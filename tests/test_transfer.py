"""A conversation moved between harnesses (`relay transfer`): each harness's record read into the
standard space, and written from it as the other's. The recorded sessions (tests/sessions) are
what Claude Code 2.1.309 and Codex 0.165 wrote running every kind of call they have."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from relay.core.ir import Media
from relay.transfer import HARNESSES, Call, Message, Session, transfer
from relay.transfer.claude_code import CURRENT, project

MADE = "(Moved from {}: made there with its `{}` tool.)"  # (a call written as another kind of tool, in its result)

SESSIONS = Path(__file__).parent / "sessions"
CLAUDE, CODEX = HARNESSES["claude_code"], HARNESSES["codex"]


def calls(session: Session) -> list[Call]:
    return [call for message in session.messages for call in message.calls]


class Home(unittest.TestCase):
    """Each harness's home in a directory of the test's own."""

    def setUp(self) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)
        patcher = mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(self.root / "claude"),
                                               "CODEX_HOME": str(self.root / "codex")})
        patcher.start()
        self.addCleanup(patcher.stop)

    def rows(self, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines()]


class ReadTests(unittest.TestCase):
    def test_claude_codes_record_in_the_standard_space(self) -> None:
        session = CLAUDE.read(SESSIONS / "claude-code-2.1.309.jsonl")
        self.assertEqual((session.cwd, session.model), ("/tmp/tmpl7d9rnxs/work", "claude-opus-5-5"))
        self.assertEqual([(m.role, m.text) for m in session.messages if not m.calls],
                         [("user", "Work on the files."), ("assistant", "FAKE FINAL ANSWER")])  # (its reminders gone)
        read, edit, write, failed, image, part, listed = calls(session)
        self.assertEqual((read.tool, read.args["path"], read.output), ("read", "/tmp/tmpl7d9rnxs/work/notes.txt", "CODE: amber\nline two\n"))
        self.assertEqual(edit.args["files"], [{"path": "/tmp/tmpl7d9rnxs/work/notes.txt", "kind": "update",
                                               "edits": [("amber", "birch", False)],
                                               "hunks": [["-CODE: amber", "+CODE: birch", " line two"]], "starts": [1]}])
        self.assertEqual(write.args["files"][0] | {"path": ""}, {"path": "", "kind": "add", "content": "hello\n"})
        self.assertEqual((failed.tool, failed.exit, failed.error, failed.output), ("shell", 3, True, "out\nerr"))
        self.assertEqual((image.tool, image.media[0].mime, image.output), ("read", "image/png", ""))
        self.assertEqual((part.args["start"], part.args["lines"], part.output), (2, 1, "line two"))
        self.assertEqual((listed.args, listed.exit, listed.output), ({"command": "ls"}, 0, "new.txt\nnotes.txt\npic.png"))

    def test_codexs_record_in_the_standard_space(self) -> None:
        session = CODEX.read(SESSIONS / "codex-0.165.0.jsonl")
        self.assertEqual((session.cwd, session.model), ("/tmp/tmp67v19v6e/work", "gpt-5.1-codex"))
        self.assertEqual([(m.role, m.text) for m in session.messages if not m.calls],  # (its own context gone)
                         [("user", "Work on the files."), ("assistant", "FAKE FINAL ANSWER")])
        read, patched, failed, image, listed = calls(session)
        self.assertEqual((read.tool, read.args, read.exit, read.output), ("shell", {"command": "cat notes.txt"}, 0, "CODE: amber\nline two\n"))
        self.assertEqual(patched.args["files"], [{"path": "notes.txt", "kind": "update", "hunks": [["-CODE: amber", "+CODE: birch"]]},
                                                 {"path": "added.txt", "kind": "add", "content": "new file\n"}])
        self.assertEqual(patched.output, "Success. Updated the following files:\nA added.txt\nM notes.txt\n")
        self.assertEqual((failed.exit, failed.error, failed.output), (3, True, "out\nerr\n"))
        self.assertEqual((image.tool, image.name, image.args["path"], image.media[0].mime), ("read", "view_image", "pic.png", "image/png"))
        self.assertEqual(listed.output, "added.txt\nnotes.txt\npic.png\n")


class WriteTests(Home):
    def test_codexs_session_as_claude_codes_own(self) -> None:
        source, written = transfer("codex", str(SESSIONS / "codex-0.165.0.jsonl"), "claude_code")
        self.assertEqual(written.path.parent, project("/tmp/tmp67v19v6e/work"))
        self.assertEqual(written.command, f"cd /tmp/tmp67v19v6e/work && claude --resume {written.id}")
        rows = self.rows(written.path)
        self.assertEqual([r["parentUuid"] for r in rows[1:]], [r["uuid"] for r in rows[:-1]])  # (one chain)
        uses = [b for r in rows if r["type"] == "assistant" for b in r["message"]["content"] if b["type"] == "tool_use"]
        self.assertEqual([u["name"] for u in uses], ["Bash", "Edit", "Write", "Bash", "Read", "Bash"])
        self.assertEqual(uses[1]["input"], {"file_path": "/tmp/tmp67v19v6e/work/notes.txt", "old_string": "CODE: amber",
                                            "new_string": "CODE: birch", "replace_all": False})
        results = {b["tool_use_id"]: (b, r["toolUseResult"]) for r in rows if r["type"] == "user"
                   for b in r["message"]["content"] if isinstance(b, dict) and b.get("type") == "tool_result"}
        self.assertEqual(results[uses[1]["id"]][0]["content"], f"The file /tmp/tmp67v19v6e/work/notes.txt has been updated successfully.{CURRENT}"
                                                               f"\n{MADE.format('Codex', 'exec_command')}")
        self.assertEqual(results[uses[2]["id"]][0]["content"], f"File created successfully at: /tmp/tmp67v19v6e/work/added.txt{CURRENT}"
                                                               f"\n{MADE.format('Codex', 'exec_command')}")
        self.assertEqual(results[uses[0]["id"]][0]["content"], "CODE: amber\nline two\n")  # (a command as a command: nothing more)
        # It starts with its own kind of note, which the model reads and its interface does not show.
        self.assertEqual((rows[0]["type"], rows[0]["isMeta"]), ("user", True))
        self.assertEqual(rows[0]["message"]["content"], (
            "<system-reminder>\nThe conversation that follows was moved here from Codex (its model: gpt-5.1-codex) with `relay transfer`. "
            "Its tool calls are written as Claude Code's own: commands as Bash, file reads as Read, file changes as Edit and Write; "
            "calls with no tool like them here are described in text. Tools it used in Codex are not available here "
            "(`exec_command`, `view_image`): use the tools you have now.\n</system-reminder>"))
        self.assertEqual(results[uses[3]["id"]][0], {"type": "tool_result", "tool_use_id": uses[3]["id"],
                                                     "content": "Exit code 3\nout\nerr\n", "is_error": True})
        self.assertEqual(results[uses[4]["id"]][0]["content"][0]["type"], "image")
        self.assertEqual(results[uses[1]["id"]][1]["structuredPatch"],  # (its interface shows the change by it)
                         [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1, "lines": ["-CODE: amber", "+CODE: birch"]}])
        self.assertFalse(any("model" in r["message"] for r in rows if r["type"] == "assistant"))  # (Codex's, not one of its own)
        # The edit and the file added, one patch in Codex, are one response of Claude Code's: its calls together.
        self.assertEqual([len(m.calls) for m in CLAUDE.read(written.path).messages if m.calls], [1, 2, 1, 1, 1])
        # Read back, the same conversation.
        again = CLAUDE.read(written.path)
        self.assertEqual([(c.tool, c.output, c.exit) for c in calls(again) if c.tool == "shell"],
                         [(c.tool, c.output, c.exit) for c in calls(source) if c.tool == "shell"])
        self.assertEqual([m.text for m in again.messages], [m.text for m in source.messages if m.text or m.calls])

    def test_claude_codes_session_as_codexs_own(self) -> None:
        source, written = transfer("claude_code", str(SESSIONS / "claude-code-2.1.309.jsonl"), "codex")
        self.assertRegex(written.path.name, rf"^rollout-\d{{4}}-\d\d-\d\dT\d\d-\d\d-\d\d-{written.id}\.jsonl$")
        self.assertTrue(written.path.is_relative_to(self.root / "codex" / "sessions"))
        rows = self.rows(written.path)
        self.assertEqual(rows[0]["payload"] | {"timestamp": "", "context_window": {}}, {
            "id": written.id, "session_id": written.id, "timestamp": "", "cwd": "/tmp/tmpl7d9rnxs/work",
            "originator": "codex_cli_rs", "cli_version": "0.165.0", "source": "cli", "model_provider": "openai",
            "history_mode": "paginated", "context_window": {}})  # (paginated: its interface shows the turns' items)
        self.assertEqual([r["ordinal"] for r in rows], list(range(len(rows))))
        items = [r["payload"] for r in rows if r["type"] == "response_item"]
        made = [(i["name"], json.loads(i["arguments"])) for i in items if i["type"] == "function_call"]
        self.assertEqual(made[0], ("exec_command", {"cmd": "cat notes.txt"}))
        self.assertEqual(made[1], ("exec_command", {"cmd": "apply_patch <<'EOF'\n*** Begin Patch\n*** Update File: notes.txt\n@@\n"
                                                          "-CODE: amber\n+CODE: birch\n line two\n*** End Patch\nEOF"}))
        self.assertEqual(made[2][1]["cmd"], "apply_patch <<'EOF'\n*** Begin Patch\n*** Add File: new.txt\n+hello\n*** End Patch\nEOF")
        self.assertEqual([name for name, _ in made[3:]], ["exec_command", "view_image", "exec_command", "exec_command"])
        self.assertEqual(made[5][1], {"cmd": "sed -n '2,2p' notes.txt"})
        outputs = [i["output"] for i in items if i["type"] == "function_call_output"]
        self.assertRegex(outputs[0], r"^Chunk ID: [0-9a-f]{6}\nWall time: 0.0000 seconds\nProcess exited with code 0\n"
                                     r"Original token count: \d+\nOutput:\nCODE: amber\nline two\n$")
        self.assertTrue(outputs[1].endswith("Exit code: 0\nWall time: 0 seconds\nOutput:\nSuccess. Updated the following files:\nM notes.txt\n"
                                            + MADE.format("Claude Code", "Edit")))
        self.assertEqual(items[0], {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": (
            "The conversation that follows was moved here from Claude Code (its model: claude-opus-5-5) with `relay transfer`. Its tool "
            "calls are written as Codex's own: commands, file reads (as cat or sed) and file changes (as apply_patch) as "
            "exec_command calls, images as view_image; calls with no tool like them here are described in text. Tools it used in "
            "Claude Code are not available here (`Bash`, `Edit`, `Read`, `Write`): use the tools you have now.")}]})
        self.assertIn("Process exited with code 3\n", outputs[3])
        self.assertEqual(outputs[4][0]["type"], "input_image")
        events = [r["payload"] for r in rows if r["type"] == "event_msg"]  # (what its interface shows: turns of items)
        self.assertEqual([e["type"] for e in events][:2] + [events[-1]["type"]], ["task_started", "item_completed", "task_complete"])
        shown = [e["item"]["type"] for e in events if e["type"] == "item_completed"]
        self.assertEqual(shown, ["UserMessage", "CommandExecution", "FileChange", "FileChange", "CommandExecution", "ImageView",
                                 "CommandExecution", "CommandExecution", "AgentMessage"])
        diff = next(e["item"]["changes"] for e in events if e["type"] == "item_completed" and e["item"]["type"] == "FileChange")
        self.assertEqual(diff, {"/tmp/tmpl7d9rnxs/work/notes.txt": {"type": "update", "move_path": None, "unified_diff":
                                "@@ -1,2 +1,2 @@\n-CODE: amber\n+CODE: birch\n line two\n"}})
        # Read back, the same conversation (Claude Code's reads, Codex's commands that read).
        again = CODEX.read(written.path)
        self.assertEqual([c.output for c in calls(again) if c.tool == "shell"],
                         [c.output for c in calls(source) if c.tool == "shell" or c.tool == "read" and not c.media])
        self.assertEqual([(c.tool, c.args["files"][0]["kind"]) for c in calls(again) if c.tool == "change"], [("change", "update"), ("change", "add")])

    def test_a_call_without_a_counterpart_is_said_as_text(self) -> None:
        agent = Call("", {}, "Agent", {"prompt": "find the parser"}, "It is in parser.py.")
        session = Session("claude_code", "s", "/w", (Message("user", "go"), Message("assistant", "Asking.", calls=(agent,))))
        written = CODEX.write(session, "/w")
        texts = [i["content"][0]["text"] for i in (r["payload"] for r in self.rows(written.path) if r["type"] == "response_item")
                 if i["type"] == "message" and i["role"] == "assistant"]
        self.assertEqual(written.notes, 1)
        self.assertEqual(len(texts), 1)
        self.assertIn("Asking.", texts[0])
        self.assertIn('[Moved from Claude Code: there I called its tool `Agent`, which is not available here, with: '
                      '{"prompt": "find the parser"}', texts[0])
        self.assertIn("It is in parser.py.", texts[0])

    def test_files_changed_as_each_harness_changes_them(self) -> None:
        change = Call("change", {"files": [
            {"path": "/w/old.txt", "kind": "update", "to": "/w/new.txt", "hunks": [[" keep", "-gone", "+came"]]},
            {"path": "/w/drop.txt", "kind": "delete"},
            {"path": "/w/over.txt", "kind": "update", "content": "all new\n"}]}, "apply_patch", "...", "done")
        session = Session("codex", "s", "/w", (Message("user", "go"), Message("assistant", calls=(change,))))
        uses = [b for r in self.rows(CLAUDE.write(session, "/w").path) if r["type"] == "assistant"
                for b in r["message"]["content"] if b["type"] == "tool_use"]
        self.assertEqual([(u["name"], u["input"].get("command") or u["input"].get("old_string")) for u in uses],
                         [("Bash", "mv /w/old.txt /w/new.txt"), ("Edit", "keep\ngone"), ("Bash", "rm /w/drop.txt"), ("Write", None)])
        items = [r["payload"] for r in self.rows(CODEX.write(session, "/w").path) if r["type"] == "response_item"]
        commands = [json.loads(i["arguments"])["cmd"] for i in items if i["type"] == "function_call"]
        self.assertEqual(commands, ["apply_patch <<'EOF'\n*** Begin Patch\n*** Update File: old.txt\n*** Move to: new.txt\n@@\n keep\n"
                                    "-gone\n+came\n*** Delete File: drop.txt\n*** End Patch\nEOF",
                                    "cat > over.txt <<'EOF'\nall new\nEOF"])


class EdgeTests(Home):
    def claude(self, *entries: dict[str, Any]) -> Path:
        path, parent = self.root / "c.jsonl", None
        rows = []
        for n, entry in enumerate(entries):
            rows.append({"parentUuid": parent, "uuid": f"u{n}", "sessionId": "s", "cwd": "/w", "timestamp": "", **entry})
            parent = None if entry.get("type") == "system" else f"u{n}"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def test_claude_codes_compaction_split_messages_and_calls_left_unanswered(self) -> None:
        use = lambda i, name="Bash": {"type": "tool_use", "id": i, "name": name, "input": {"command": f"echo {i}"}}  # noqa: E731
        result = lambda i: {"type": "tool_result", "tool_use_id": i, "content": i}  # noqa: E731
        path = self.claude(
            {"type": "user", "message": {"role": "user", "content": "before the compaction"}},
            {"type": "system", "subtype": "compact_boundary"},  # (a new start: what came before is the summary)
            {"type": "user", "isCompactSummary": True, "message": {"role": "user", "content":
                "This session is being continued from a previous conversation that ran out of context. SUMMARY"}},
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>\nAs you answer, you can use this context\n</system-reminder>"},
                {"type": "text", "text": "go on"}]}},
            {"type": "assistant", "message": {"id": "m1", "role": "assistant", "content": [{"type": "thinking", "thinking": "hm", "signature": "x"}]}},
            {"type": "assistant", "message": {"id": "m1", "role": "assistant", "content": [use("a")]}},
            {"type": "assistant", "message": {"id": "m1", "role": "assistant", "content": [use("b")]}},
            {"type": "user", "message": {"role": "user", "content": [result("b")]}},
            {"type": "user", "message": {"role": "user", "content": [result("a")]}},
            {"type": "attachment", "attachment": {"type": "todo"}},
            {"type": "assistant", "isSidechain": True, "message": {"id": "x", "role": "assistant", "content": [{"type": "text", "text": "a sub-agent"}]}},
            {"type": "assistant", "message": {"id": "m2", "role": "assistant", "content": [{"type": "text", "text": "and"}, use("c")]}})
        messages = CLAUDE.read(path).messages
        self.assertEqual([(m.role, m.text[:44]) for m in messages],
                         [("user", "This session is being continued from a previ"), ("user", "go on"), ("assistant", ""), ("assistant", "and")])
        self.assertEqual([c.args["command"] for c in messages[2].calls], ["echo a", "echo b"])  # (one response, its calls in order)
        self.assertEqual(messages[3].calls, ())  # (the session stopped before its call was answered)

    def test_codexs_compaction_and_calls_of_one_response(self) -> None:
        def item(payload: dict[str, Any]) -> dict[str, Any]:
            return {"timestamp": "", "type": "response_item", "payload": payload}
        call = lambda i: item({"type": "function_call", "name": "exec_command", "arguments": json.dumps({"cmd": f"echo {i}"}), "call_id": i})  # noqa: E731
        output = lambda i: item({"type": "function_call_output", "call_id": i, "output": f"Process exited with code 0\nOutput:\n{i}\n"})  # noqa: E731
        user = lambda text: item({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})  # noqa: E731
        rows = [{"type": "session_meta", "payload": {"id": "s", "cwd": "/w"}}, user("old task"), call("old"), output("old"),
                {"type": "compacted", "payload": {"message": "SUMMARY", "replacement_history": [
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "old task"}]}]}},
                user("<environment_context>\n  <cwd>/w</cwd>\n</environment_context>"), user("new task"),
                item({"type": "reasoning", "summary": [], "encrypted_content": "e"}),
                item({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Two at once."}]}),
                call("a"), call("b"), output("a"), output("b"), call("c"), output("c"),
                item({"type": "web_search_call", "action": {"type": "search", "query": "parser"}})]
        path = self.root / "rollout.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        messages = CODEX.read(path).messages
        self.assertEqual([(m.role, m.text, [c.name for c in m.calls]) for m in messages],
                         [("user", "old task", []), ("user", "new task", []), ("assistant", "Two at once.", ["exec_command"] * 2),
                          ("assistant", "", ["exec_command"]), ("assistant", "", ["web_search_call"])])
        self.assertEqual(messages[2].calls[1].output, "b\n")

    def test_codexs_code_mode_scripts_that_make_one_call(self) -> None:
        # (as Codex 0.164 recorded them with gpt-6-luna: its one tool, `exec`, runs a script)
        def script(n: int, code: str, *output: str) -> list[dict[str, Any]]:
            return [{"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec", "input": code, "call_id": f"c{n}"}},
                    {"type": "response_item", "payload": {"type": "custom_tool_call_output", "call_id": f"c{n}", "output": [
                        {"type": "input_text", "text": text} for text in ("Script completed\nWall time 0.1 seconds\nOutput:\n", *output)]}}]
        rows = [{"type": "session_meta", "payload": {"id": "s", "cwd": "/project"}},
                {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}},
                *script(1, 'const r = await tools.exec_command({cmd:"cat file_1.txt",workdir:"/project",max_output_tokens:3000});\n'
                           "text(r.output);\n", "file 1 text\n"),
                *script(2, "await tools.apply_patch(`*** Begin Patch\n*** Add File: a.txt\n+a $HOME\n*** End Patch`);\n", "Done"),
                *script(3, 'for (const f of ["a", "b"]) { text((await tools.exec_command({cmd: `cat ${f}`})).output); }', "a\nb\n")]
        path = self.root / "rollout.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        shell, patch, loop = calls(CODEX.read(path))
        self.assertEqual((shell.tool, shell.args, shell.output), ("shell", {"command": "cat file_1.txt"}, "file 1 text\n"))
        self.assertEqual(patch.args["files"], [{"path": "a.txt", "kind": "add", "content": "a $HOME\n"}])
        self.assertEqual((loop.tool, loop.output), ("", "a\nb\n"))  # (a script of its own: said as text)

    def test_claude_codes_folder_for_a_directory_is_its_own(self) -> None:
        self.assertEqual(project("/tmp/x y/work").name, "-tmp-x-y-work")
        long = "/work/" + "deep-folder_name." * 14 + "/项目/end"  # (Claude Code's own, computed by its code)
        self.assertEqual(project(long).name, "-work-" + "deep-folder-name-" * 11 + "deep-fo-92b1b3")

    def test_media_keep_their_place(self) -> None:
        image = Media("image", "image/png", "aGk=")
        session = Session("codex", "s", "/w", (Message("user", "look", media=(image,)),))
        rows = self.rows(CLAUDE.write(session, "/w").path)  # (after its note of where the conversation came from)
        self.assertEqual(rows[1]["message"]["content"], [{"type": "text", "text": "look"}, {
            "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}])
        items = [r["payload"] for r in self.rows(CODEX.write(session, "/w").path) if r["type"] == "response_item"]
        self.assertEqual(items[0]["content"][1], {"type": "input_image", "image_url": "data:image/png;base64,aGk="})  # (its own: no note)


if __name__ == "__main__":
    unittest.main()
