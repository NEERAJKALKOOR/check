from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_FOLDER_MAP: dict[str, list[str]] = {
    "images": [".jpg", ".jpeg", ".png", ".gif"],
    "docs": [".pdf", ".docx", ".txt"],
    "videos": [".mp4", ".mkv", ".mov"],
    "audio": [".mp3", ".wav"],
    "code": [".py", ".js", ".cpp", ".java"],
    "archives": [".zip", ".rar", ".tar"],
}
VALID_CATEGORIES = tuple(DEFAULT_FOLDER_MAP.keys()) + ("others",)


@dataclass(frozen=True)
class PlannedMove:
    source: Path
    target: Path
    category: str


@dataclass(frozen=True)
class ActionResult:
    source: Path
    target: Path
    category: str
    moved: bool


class FileScanner:
    def scan(self, folder: Path) -> list[Path]:
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")
        return [
            entry
            for entry in folder.iterdir()
            if entry.is_file() and entry.name != "organizer.log"
        ]


class FileClassifier:
    def __init__(
        self,
        folder_map: dict[str, list[str]] | None = None,
        llm_classifier: Callable[[str], str] | None = None,
    ) -> None:
        self.folder_map = folder_map or DEFAULT_FOLDER_MAP
        self.llm_classifier = llm_classifier

    def classify(self, file_path: Path) -> str:
        if self.llm_classifier is not None:
            llm_category = self.llm_classifier(file_path.name).strip().lower()
            if llm_category in VALID_CATEGORIES:
                return llm_category
        extension = file_path.suffix.lower()
        for category, extensions in self.folder_map.items():
            if extension in extensions:
                return category
        return "others"


class ActionPlanner:
    def __init__(self, classifier: FileClassifier) -> None:
        self.classifier = classifier

    def plan(self, files: Iterable[Path], base_folder: Path) -> list[PlannedMove]:
        planned_moves: list[PlannedMove] = []
        for file_path in files:
            category = self.classifier.classify(file_path)
            planned_moves.append(
                PlannedMove(
                    source=file_path,
                    target=base_folder / category / file_path.name,
                    category=category,
                )
            )
        return planned_moves


class ExecutorAgent:
    def execute(self, planned_moves: Iterable[PlannedMove], dry_run: bool) -> list[ActionResult]:
        results: list[ActionResult] = []
        for move in planned_moves:
            final_target = self._resolve_target_collision(move.target)
            if dry_run:
                results.append(
                    ActionResult(
                        source=move.source,
                        target=final_target,
                        category=move.category,
                        moved=False,
                    )
                )
                continue

            final_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(move.source), str(final_target))
            results.append(
                ActionResult(
                    source=move.source,
                    target=final_target,
                    category=move.category,
                    moved=True,
                )
            )
        return results

    @staticmethod
    def _resolve_target_collision(target: Path) -> Path:
        if not target.exists():
            return target

        stem = target.stem
        suffix = target.suffix
        index = 1
        while True:
            candidate = target.with_name(f"{stem}_{index}{suffix}")
            if not candidate.exists():
                return candidate
            index += 1


class ActionLogger:
    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file

    def log(self, action: ActionResult) -> None:
        timestamp = dt.datetime.now().isoformat()
        status = "MOVED" if action.moved else "DRY_RUN"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"{timestamp} - {status} - {action.source.name} -> "
                f"{action.category}/ ({action.target.name})\n"
            )


class FileOrganizerAgent:
    def __init__(
        self,
        scanner: FileScanner,
        planner: ActionPlanner,
        executor: ExecutorAgent,
        logger: ActionLogger,
    ) -> None:
        self.scanner = scanner
        self.planner = planner
        self.executor = executor
        self.logger = logger

    def organize(self, folder: Path, dry_run: bool) -> list[ActionResult]:
        files = self.scanner.scan(folder)
        plan = self.planner.plan(files, folder)
        results = self.executor.execute(plan, dry_run=dry_run)
        for result in results:
            self.logger.log(result)
        return results


def classify_with_ollama(
    filename: str,
    *,
    model: str = "qwen2.5-coder",
    endpoint: str = "http://localhost:11434/api/generate",
) -> str:
    prompt = (
        "You are a file organization agent.\n"
        "Classify this file into one category: "
        "images, docs, videos, audio, code, archives, others.\n\n"
        f"File name: {filename}\n\n"
        "Return only one word."
    )

    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return "others"

    return str(body.get("response", "others")).strip().lower()


def build_agent(base_folder: Path, use_llm: bool, llm_model: str, llm_endpoint: str) -> FileOrganizerAgent:
    llm_classifier = None
    if use_llm:
        llm_classifier = lambda filename: classify_with_ollama(
            filename,
            model=llm_model,
            endpoint=llm_endpoint,
        )

    scanner = FileScanner()
    classifier = FileClassifier(llm_classifier=llm_classifier)
    planner = ActionPlanner(classifier)
    executor = ExecutorAgent()
    logger = ActionLogger(base_folder / "organizer.log")
    return FileOrganizerAgent(scanner, planner, executor, logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize files in a folder using an agentic pipeline")
    parser.add_argument("path", type=Path, help="Folder path to organize")
    parser.add_argument("--dry-run", action="store_true", help="Plan and log changes without moving files")
    parser.add_argument("--use-llm", action="store_true", help="Use Ollama for classification")
    parser.add_argument("--llm-model", default="qwen2.5-coder", help="Ollama model name")
    parser.add_argument("--llm-endpoint", default="http://localhost:11434/api/generate", help="Ollama API endpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = args.path.resolve()
    agent = build_agent(
        base_folder=folder,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        llm_endpoint=args.llm_endpoint,
    )

    try:
        results = agent.organize(folder=folder, dry_run=args.dry_run)
    except (FileNotFoundError, NotADirectoryError) as error:
        print(error)
        return 1

    for result in results:
        status = "[DRY RUN] Would move" if not result.moved else "Moved"
        print(f"{status} {result.source.name} -> {result.category}/{result.target.name}")

    if not results:
        print("No files to organize.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
