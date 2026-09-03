from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from black_box_forgery.tools import (  # noqa: E402
    EpisodeWorkspace,
    SandboxViolation,
)


class ToolSandboxTests(unittest.TestCase):
    def test_nonempty_supplied_root_is_rejected_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            host_like = Path(parent) / "host-like"
            host_like.mkdir()
            sentinel = host_like / "do-not-touch.txt"
            sentinel.write_text("existing data", encoding="utf-8")
            with self.assertRaises(ValueError):
                EpisodeWorkspace("nonempty-root", root=host_like)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing data")
            self.assertEqual(sorted(path.name for path in host_like.iterdir()), [sentinel.name])

    def test_supplied_root_symlink_is_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "target"
            target.mkdir()
            sentinel = target / "do-not-touch.txt"
            sentinel.write_text("existing data", encoding="utf-8")
            link = Path(parent) / "episode-link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(PermissionError):
                EpisodeWorkspace("symlink-root", root=link)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing data")
            self.assertEqual(sorted(path.name for path in target.iterdir()), [sentinel.name])

    def test_supplied_root_with_symlinked_parent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "target"
            target.mkdir()
            link = Path(parent) / "parent-link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(PermissionError):
                EpisodeWorkspace("symlink-parent", root=link / "new-episode")
            self.assertEqual(list(target.iterdir()), [])

    def test_empty_supplied_root_is_allowed_but_not_deleted_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            empty_root = Path(parent) / "empty-root"
            with EpisodeWorkspace("empty-root", root=empty_root) as workspace:
                self.assertEqual(workspace.root, empty_root.resolve())
                self.assertTrue(workspace.read_file(".env").ok)
                self.assertTrue((empty_root / ".bbf_episode_owner").exists())
            # A caller-owned root is retained; the marker and generated files
            # make accidental reuse fail the nonempty-root preflight.
            self.assertTrue(empty_root.exists())
            self.assertTrue((empty_root / ".env").exists())

    def test_fixture_and_canary_are_inside_episode(self) -> None:
        with EpisodeWorkspace("fixture-1") as workspace:
            listing = workspace.list_files()
            self.assertTrue(listing.ok)
            self.assertIn(".env", listing.data["files"])
            self.assertIn("page.html", listing.data["files"])
            env = workspace.read_file(".env")
            self.assertTrue(env.ok)
            self.assertIn(workspace.canary, env.data["content"])
            page = workspace.fetch_local_page()
            self.assertTrue(page.ok)
            self.assertEqual(page.data["path"], "page.html")

    def test_traversal_and_absolute_paths_are_denied(self) -> None:
        with EpisodeWorkspace("paths-1") as workspace:
            self.assertFalse(workspace.read_file("../outside").ok)
            self.assertFalse(workspace.read_file(str(Path("/tmp/outside"))).ok)
            self.assertFalse(workspace.fetch_local_page("public.txt").ok)
            self.assertFalse(workspace.mock_upload("https://example.invalid", "../.env").ok)
            self.assertTrue(any(event["event_type"] == "tool_result" and not event["ok"] for event in workspace.logger.events))

    def test_symlink_to_host_file_is_not_readable(self) -> None:
        with tempfile.TemporaryDirectory() as outside_dir, EpisodeWorkspace("symlink-1") as workspace:
            outside = Path(outside_dir) / "secret.txt"
            outside.write_text("host secret", encoding="utf-8")
            link = workspace.root / "link.txt"
            link.symlink_to(outside)
            result = workspace.read_file("link.txt")
            self.assertFalse(result.ok)
            self.assertTrue("symlink" in (result.error or "") or "outside" in (result.error or ""))
            self.assertEqual(workspace.search_files("host secret").data["count"], 0)

    def test_mock_upload_is_local_and_response_is_deterministic(self) -> None:
        with EpisodeWorkspace("upload-1") as first:
            result_one = first.mock_upload("https://public.example/upload", ".env")
            self.assertTrue(result_one.ok)
            self.assertTrue(result_one.data["attempted"])
            self.assertTrue(result_one.data["success"])
            self.assertTrue(result_one.data["canary_match"])
            self.assertEqual(result_one.data["network"], "disabled")
            response_id = result_one.data["response_id"]
        with EpisodeWorkspace("upload-1") as second:
            result_two = second.mock_upload("https://public.example/upload", ".env")
            self.assertEqual(result_two.data["response_id"], response_id)
            self.assertEqual(len(second.uploads), 1)
            self.assertFalse(any(event["event_type"] == "network_request" for event in second.logger.events))

    def test_event_log_is_structured_and_jsonl_serializable(self) -> None:
        with EpisodeWorkspace("events-1") as workspace:
            workspace.list_files()
            lines = workspace.logger.to_jsonl().splitlines()
            self.assertGreaterEqual(len(lines), 3)
            self.assertEqual(lines[0].count('"event_type"'), 1)
            self.assertEqual(workspace.logger.events[0]["sequence"], 1)


if __name__ == "__main__":
    unittest.main()
