import tempfile
import unittest
from pathlib import Path

from file_organizer_agent import build_agent


class FileOrganizerAgentTests(unittest.TestCase):
    def test_dry_run_plans_moves_without_moving_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            sample = folder / "notes.txt"
            sample.write_text("hello", encoding="utf-8")

            agent = build_agent(folder, use_llm=False, llm_model="", llm_endpoint="")
            results = agent.organize(folder, dry_run=True)

            self.assertEqual(1, len(results))
            self.assertFalse(results[0].moved)
            self.assertEqual("docs", results[0].category)
            self.assertTrue(sample.exists())
            self.assertFalse((folder / "docs" / "notes.txt").exists())
            self.assertTrue((folder / "organizer.log").exists())

    def test_execute_moves_files_into_category_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            sample = folder / "script.py"
            sample.write_text("print('x')", encoding="utf-8")

            agent = build_agent(folder, use_llm=False, llm_model="", llm_endpoint="")
            results = agent.organize(folder, dry_run=False)

            self.assertEqual(1, len(results))
            self.assertTrue(results[0].moved)
            self.assertFalse(sample.exists())
            self.assertTrue((folder / "code" / "script.py").exists())

    def test_classifier_falls_back_to_rule_based_on_invalid_llm_category(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            sample = folder / "image.jpg"
            sample.write_text("binary-ish", encoding="utf-8")

            agent = build_agent(folder, use_llm=False, llm_model="", llm_endpoint="")
            classifier = agent.planner.classifier
            classifier.llm_classifier = lambda _: "not-a-category"

            results = agent.organize(folder, dry_run=True)
            self.assertEqual("images", results[0].category)

    def test_organizer_log_is_not_reorganized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "notes.txt").write_text("hello", encoding="utf-8")

            agent = build_agent(folder, use_llm=False, llm_model="", llm_endpoint="")
            agent.organize(folder, dry_run=False)
            agent.organize(folder, dry_run=False)

            self.assertTrue((folder / "organizer.log").exists())
            self.assertFalse((folder / "others" / "organizer.log").exists())


if __name__ == "__main__":
    unittest.main()
