"""The media contract, protocol by protocol: a codec reports all of an item's non-text content and writes
back exactly the media it is given, each as it came; core counts them and shows them; strategies only choose."""

from __future__ import annotations

import base64
import json
import struct
import unittest
from typing import Any

from relay.core.engine import Engine
from relay.core.ir import Context, Item, Kind, Media
from relay.core.media import counted, kept, lines, marks
from relay.core.tokens import AUDIO_BYTES_PER_TOKEN, FILE_BYTES_PER_TOKEN, IMAGE_TOKENS, item_tokens, media_tokens
from relay.core.truncation import truncate_request
from relay.harnesses import Harness
from relay.protocols import AnthropicMessages, Gemini, OpenAIChat, OpenAIResponses
from relay.strategies.jev import _described

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 750, 100) + b"\x08\x02\x00\x00\x00").decode()
PDF = base64.b64encode(b"%PDF-1.7 report" * 40).decode()
WAV = base64.b64encode(b"RIFFwave" * 40).decode()
URL = f"data:image/png;base64,{PNG}"
PDF_URL = f"data:application/pdf;base64,{PDF}"
RESPONSE = {"name": "shot", "response": {"output": "screen"}}
PARTS = (
    (OpenAIResponses(), {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "look"}, {"type": "input_image", "image_url": URL, "detail": "original"},
        {"type": "input_file", "file_data": PDF_URL, "filename": "report.pdf"},
        {"type": "input_audio", "input_audio": {"data": WAV, "format": "wav"}},
        {"type": "input_widget", "widget": "w7"}]}, ["image", "file", "audio", "input_widget"], ["original", "report.pdf", "w7"]),
    (OpenAIResponses(), {"type": "custom_tool_call_output", "call_id": "c", "output": [
        {"type": "input_text", "text": "look"}, {"type": "input_image", "image_url": URL, "detail": "original"}]}, ["image"], ["original"]),
    (OpenAIChat(), {"role": "user", "content": [
        {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": URL, "detail": "low"}},
        {"type": "file", "file": {"file_data": PDF_URL, "filename": "report.pdf"}},
        {"type": "input_audio", "input_audio": {"data": WAV, "format": "wav"}},
        {"type": "video_url", "video_url": "v7"}]}, ["image", "file", "audio", "video_url"], ['"detail": "low"', "report.pdf", "v7"]),
    (OpenAIChat(), {"role": "tool", "tool_call_id": "t", "content": [
        {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": URL, "detail": "low"}}]}, ["image"], ["low"]),
    (AnthropicMessages(), {"role": "user", "content": [
        {"type": "text", "text": "look"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
        {"type": "document", "title": "report", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF}},
        {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "plain d7"}},
        {"type": "search_result", "title": "s7", "source": "u", "content": []}]},
     ["image", "file", "document", "search_result"], ['"title": "report"', "plain d7", "s7"]),
    (AnthropicMessages(), {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": [
        {"type": "text", "text": "look"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}]}]},
     ["image"], []),
    (Gemini(), {"role": "user", "parts": [
        {"text": "look"}, {"inlineData": {"mimeType": "image/png", "data": PNG, "displayName": "d7"}, "mediaResolution": "HIGH"},
        {"fileData": {"mimeType": "application/pdf", "fileUri": "gs://b/f.pdf"}}]}, ["image", "file"], ["d7", "HIGH", "gs://b/f.pdf"]),
    (Gemini(), {"role": "user", "parts": [{"functionResponse": RESPONSE}, {"inlineData": {"mimeType": "image/png", "data": PNG}}]},
     ["image"], []),
    (Gemini(), {"role": "user", "parts": [{"functionResponse": {**RESPONSE, "parts": [
        {"inlineData": {"mimeType": "image/png", "data": PNG, "displayName": "d7"}}]}}]}, ["image"], ["d7"]),
)


class EveryProtocolTests(unittest.TestCase):
    def test_all_of_an_items_non_text_content_is_its_media(self) -> None:
        for codec, wire, types, _ in PARTS:
            with self.subTest(codec=codec.name, types=types):
                item = codec.classify(wire)
                self.assertEqual([m.type for m in item.media], types)
                self.assertNotIn(PNG, item.text)  # (and none of it is text)

    def test_media_kept_through_an_edit_go_back_as_they_came(self) -> None:
        for codec, wire, types, fields in PARTS:
            with self.subTest(codec=codec.name, types=types):
                item = codec.classify(wire)
                edited = codec.edit(wire, "new text", item.media)
                self.assertEqual((codec.classify(edited).text, codec.classify(edited).media), ("new text", item.media))
                for field in fields:
                    self.assertIn(field, json.dumps(edited))

    def test_media_taken_out_leave_nothing_behind(self) -> None:
        for codec, wire, types, fields in PARTS:
            with self.subTest(codec=codec.name, types=types):
                edited = json.dumps(codec.edit(wire, "new text", ()))
                self.assertEqual(codec.classify(json.loads(edited)).media, ())
                for payload in (PNG, PDF, WAV, *fields):
                    self.assertNotIn(payload, edited)

    def test_one_taken_out_leaves_the_others(self) -> None:
        for codec, wire, types, _ in PARTS:
            if len(types) > 1:
                with self.subTest(codec=codec.name, types=types):
                    item = codec.classify(wire)
                    edited = codec.edit(wire, item.text, item.media[1:])
                    self.assertEqual([m.type for m in codec.classify(edited).media], types[1:])
                    self.assertNotIn(PNG, json.dumps(edited))

    def test_media_counts_and_its_base64_does_not_count_as_text(self) -> None:
        for codec, wire, types, _ in PARTS:
            with self.subTest(codec=codec.name, types=types):
                item = codec.classify(wire)
                self.assertTrue(all(media_tokens(m) > 0 for m in item.media))
                self.assertLess(item_tokens(item), sum(map(media_tokens, item.media)) + len(item.text) + 16)

    def test_what_each_kind_costs(self) -> None:
        self.assertEqual(media_tokens(Media("image", "image/png", PNG)), 100)  # (750 by 100 pixels)
        self.assertEqual(media_tokens(Media("file", "application/pdf", "A" * 1600)), 1200 // FILE_BYTES_PER_TOKEN)
        self.assertEqual(media_tokens(Media("audio", "audio/wav", "A" * 4000)), 3000 // AUDIO_BYTES_PER_TOKEN)
        self.assertEqual(media_tokens(Media("file", url="file-7")), IMAGE_TOKENS)  # (its size unknown)


class TextViewTests(unittest.TestCase):
    item = Item(Kind.TOOL_RESULT, "the page", media=(Media("image", "image/png", PNG), Media("file", "application/pdf", PDF),
                                                     Media("audio", "audio/wav", WAV)))

    def test_every_kind_is_a_line_and_a_line_gone_is_that_one_gone(self) -> None:
        shown = lines(self.item).splitlines()
        self.assertEqual([line.split(",")[0] for line in shown],
                         ["[image 1 of this turn", "[file 2 of this turn", "[audio 3 of this turn"])
        left = kept(self.item, "\n".join(["the new page", shown[0], shown[2]]))
        self.assertEqual((left.text, [m.type for m in left.media]), ("the new page", ["image", "audio"]))

    def test_marks_and_counts(self) -> None:
        self.assertEqual(marks(self.item.media, 4), ["[image 5]", "[file 6]", "[audio 7]"])
        self.assertEqual(counted((*self.item.media, self.item.media[0])), "2 images and 1 file and 1 audio")
        self.assertEqual(_described(7, self.item.media[:2]), "7 chars and 1 image and 1 file")


class Dropping:
    """Takes `take` out of every item's media, and rewrites its text where `text` says so."""

    name = "dropping"

    def __init__(self, text: str | None = None) -> None:
        self.text = text

    def fingerprint(self) -> dict[str, Any]:
        return {"name": self.name}

    def plan(self, request: Any, summarizer: Any) -> Context:
        return Context(tuple(Item(i.kind, self.text or i.text, i.ref) if i.media else i for i in request.current))


class EngineTests(unittest.TestCase):
    history = [{"type": "message", "role": "user", "content": "fix it"},
               {"type": "function_call", "call_id": "c", "name": "shot", "arguments": "{}"},
               {"type": "function_call_output", "call_id": "c", "output": [
                   {"type": "input_text", "text": "screen"}, {"type": "input_image", "image_url": URL}]}]

    def send(self, strategy: Any) -> dict[str, Any]:
        body = {"model": "m", "tools": [{"type": "function", "name": "shot", "parameters": {}}], "input": self.history}
        return Engine(strategy).prepare(OpenAIResponses(), Harness(), body, tenant="t", post=lambda r: (500, {})).body["input"][-1]

    def test_an_item_that_lost_its_media_and_nothing_else_says_so(self) -> None:
        self.assertEqual(self.send(Dropping())["output"], "screen\n[1 image removed]")

    def test_new_text_speaks_for_itself(self) -> None:
        self.assertEqual(self.send(Dropping("the save dialog"))["output"], "the save dialog")


class Counting:
    """An input budget in bytes of JSON, so that media count."""

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def limit_for(self, body: dict[str, Any]) -> int:
        return self.limit

    def count(self, codec: Any, body: dict[str, Any]) -> int:
        return len(json.dumps(body))


class TruncationTests(unittest.TestCase):
    @staticmethod
    def shot(n: int, output: str | list) -> list[dict[str, Any]]:
        return [{"type": "function_call", "call_id": f"c{n}", "name": "shot", "arguments": "{}"},
                {"type": "function_call_output", "call_id": f"c{n}", "output": output}]

    def image(self, n: int) -> list[dict[str, Any]]:
        return [{"type": "input_text", "text": f"screen {n}"}, {"type": "input_image", "image_url": f"{URL}{n:04d}" + "A" * 900}]

    def test_media_go_first_oldest_first_and_the_newest_stay(self) -> None:
        task = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix it"}, *self.image(0)[1:]]}
        input_ = [task, *self.shot(1, self.image(1)), *self.shot(2, self.image(2))]
        body = {"model": "m", "input": input_}
        limit = len(json.dumps(body)) - 1800
        trimmed, report = truncate_request(OpenAIResponses(), body, Counting(limit), scope="main")
        items = trimmed["input"]
        self.assertEqual(len(items), 5)  # (no tool call was given up: only their media)
        self.assertEqual(report["stripped_media_items"], 2)
        self.assertEqual(report["dropped_groups"], 0)
        self.assertEqual(items[0]["content"], [{"type": "input_text", "text": "fix it\n[1 image removed]"}])  # (a pinned item's too)
        self.assertEqual(items[2]["output"], "screen 1\n[1 image removed]")
        self.assertEqual(items[4], input_[4])


if __name__ == "__main__":
    unittest.main()
