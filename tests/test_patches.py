from __future__ import annotations

import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from xml.etree.ElementTree import fromstring

from epub_translator.segment import search_inline_segments, search_text_segments
from epub_translator.translation.xml_interrupter import XMLInterrupter

from core import patches


class _Template:
    def render(self, **_kwargs: object) -> str:
        return "Translate to Korean."


class _FakeLLM:
    deterministic_xml_fill = True
    translation_control = None

    def template(self, _name: str) -> _Template:
        return _Template()


class _FakeTranslator:
    _translation_llm = _FakeLLM()
    _target_language = "Korean"
    _user_prompt = ""


class _FakeContext:
    def __init__(self, response: dict[str, object] | str) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def request(self, input, **_kwargs: object) -> str:
        payload = json.loads(input[1].message)
        self.requests.append(payload)
        if isinstance(self.response, str):
            return self.response
        return json.dumps(self.response, ensure_ascii=False)


def _first_inline_segment(xml: str):
    interrupter = XMLInterrupter()
    root = fromstring(xml)
    text_segments = interrupter.interrupt_source_text_segments(search_text_segments(root))
    return next(search_inline_segments(text_segments))


class PatchTranslationTests(unittest.TestCase):
    def test_slot_translation_uses_paragraph_context_and_preserves_code_spacing(self) -> None:
        inline_segment = _first_inline_segment(
            "<p>Clients that call <code>/v1/jobs</code> must handle HTTP 500 changes.</p>"
        )
        context = _FakeContext(
            {
                "slots": [
                    {"id": "s0", "text": "호출하는 클라이언트는"},
                    {"id": "s2", "text": "HTTP 500 변경을 처리해야 한다."},
                ]
            }
        )

        mapping = patches._deterministic_inline_segment_mapping(_FakeTranslator(), inline_segment, context)

        self.assertIsNotNone(mapping)
        assert mapping is not None
        _, translated_segments = mapping
        self.assertEqual(translated_segments[0].text, "호출하는 클라이언트는 ")
        self.assertEqual(translated_segments[1].text, "__EN2KO_CODE_000000__")
        self.assertEqual(translated_segments[2].text, " HTTP 500 변경을 처리해야 한다.")
        self.assertEqual([slot["id"] for slot in context.requests[0]["slots"]], ["s0", "s2"])
        self.assertIn("__EN2KO_CODE_000000__", context.requests[0]["context"])

    def test_validator_blocks_instruction_leaks(self) -> None:
        with self.assertRaises(patches._SlotValidationError):
            patches._parse_slot_response(
                '{"slots":[{"id":"s0","text":"Please provide the text to translate."}]}',
                [patches._TranslationSlot(id="s0", index=0, source_text="Required")],
            )

    def test_failed_slot_translation_keeps_source_text(self) -> None:
        inline_segment = _first_inline_segment("<p>Required</p>")
        context = _FakeContext('{"slots":[{"id":"s0","text":"<rules> do not translate"}]}')

        mapping = patches._deterministic_inline_segment_mapping(_FakeTranslator(), inline_segment, context)

        self.assertIsNone(mapping)
        self.assertEqual(len(context.requests), 2)

    def test_angle_code_tokens_are_not_sent_to_translation(self) -> None:
        self.assertFalse(patches._has_translatable_text("<BOS>"))
        self.assertFalse(patches._has_translatable_text("</EOS>"))

    def test_zip_migration_writes_each_path_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.epub"
            target = Path(tmp) / "target.epub"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(source, "w") as zf:
                    zf.writestr("mimetype", "application/epub+zip")
                    zf.writestr("EPUB/nav.xhtml", "first")
                    zf.writestr("EPUB/nav.xhtml", "second")
                    zf.writestr("EPUB/chapter.xhtml", "chapter")

            with patches._Zip(source, target):
                pass

            with zipfile.ZipFile(target) as zf:
                names = zf.namelist()
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(names.count("EPUB/nav.xhtml"), 1)

    def test_zip_migration_preserves_dot_prefixed_asset_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.epub"
            target = Path(tmp) / "target.epub"
            with zipfile.ZipFile(source, "w") as zf:
                zf.writestr("mimetype", "application/epub+zip")
                zf.writestr("OEBPS/content.opf", "<package/>")
                zf.writestr("./OEBPS/override_v1.css", "body { color: black; }")
                zf.writestr("META-INF/com.apple.ibooks.display-options.xml", "<display_options/>")

            with patches._Zip(source, target) as zf:
                with zf.replace(Path("OEBPS/content.opf")) as target_file:
                    target_file.write(b"<package updated='true'/>")

            with zipfile.ZipFile(target) as zf:
                names = zf.namelist()
                self.assertIn("./OEBPS/override_v1.css", names)
                self.assertIn("META-INF/com.apple.ibooks.display-options.xml", names)
                self.assertNotIn("OEBPS/override_v1.css", names)
                self.assertEqual(zf.read("./OEBPS/override_v1.css"), b"body { color: black; }")


if __name__ == "__main__":
    unittest.main()
