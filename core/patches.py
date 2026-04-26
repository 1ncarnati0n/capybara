"""
Workarounds for upstream bugs in oomol-lab/epub-translator.

1. IRI 인코딩된 href vs. 원본 ZIP 엔트리
이 라이브러리는 OPF에서 챕터/TOC/매니페스트의 href를 그대로 읽어온 뒤(epub_translator/epub/spines.py 및 toc.py 참조), 
이를 zipfile.ZipFile.open(...)에 전달합니다. 
EPUB 스펙에 따르면 이러한 href는 IRI 인코딩되어야 하지만(공백 → %20, 괄호 → %28/%29 등), 
ZIP 엔트리에는 인코딩되지 않은 원본 파일명이 저장되어 있습니다. 
이를 해결하기 위해 epub_translator.epub.zip.Zip을 몽키패치하여, 
zipfile에 경로를 전달하기 전에 모든 경로를 URL 디코딩합니다.

2. XHTML 챕터 내 HTML 네임드 엔티티
xml.etree.ElementTree.fromstring은 XML 기본 내장 엔티티 5개만 인식합니다. 
그런데 XHTML 챕터에는 &nbsp;, &copy;, &mdash; 등 XHTML DTD에 선언된 엔티티가 흔하게 사용되며, 
ElementTree는 해당 DTD를 가져오지 않습니다. 
이를 해결하기 위해 epub_translator.xml.xml_like.fromstring을 몽키패치하여, 
파싱 전에 HTML5 네임드 엔티티를 해당 유니코드 문자로 치환합니다.

3. 인라인 append 표시 방식
APPEND_TEXT 모드에서 업스트림은 번역문을 동일 텍스트 노드에 공백 하나만 넣고 이어붙입니다. 
한국어 리뷰용 EPUB에서는 번역문을 다음 줄에 은은한 하이라이트와 함께 표시하되, 
REPLACE 모드는 그대로 유지하도록 변경합니다.

이 모듈을 임포트하면 위 패치들이 사이드 이펙트로 자동 적용됩니다.
"""

from __future__ import annotations

import html.entities
import re
import urllib.parse
from pathlib import Path
from xml.etree.ElementTree import Element, fromstring as _et_fromstring

from epub_translator.epub import zip as _zipmod
from epub_translator.segment import TextSegment
from epub_translator.xml import xml_like as _xml_like_mod
from epub_translator.xml import index_of_parent as _index_of_parent
from epub_translator.translation import xml_interrupter as _xml_interrupter_mod
from epub_translator.xml_translator import submitter as _submitter_mod
from epub_translator.xml_translator.submitter import SubmitKind as _SubmitKind

_Zip = _zipmod.Zip
_orig_read = _Zip.read
_orig_replace = _Zip.replace
_orig_migrate = _Zip.migrate
_orig_list_files = _Zip.list_files


def _decode(path: Path) -> Path:
    return Path(urllib.parse.unquote(path.as_posix()))


def _read(self, path: Path):
    return _orig_read(self, _decode(path))


def _replace(self, path: Path):
    return _orig_replace(self, _decode(path))


def _migrate(self, path: Path):
    return _orig_migrate(self, _decode(path))


def _list_files(self, prefix_path: Path | None = None):
    if prefix_path is not None:
        prefix_path = _decode(prefix_path)
    return _orig_list_files(self, prefix_path)


_Zip.read = _read
_Zip.replace = _replace
_Zip.migrate = _migrate
_Zip.list_files = _list_files


_NAMED_ENTITY_RE = re.compile(r"&([a-zA-Z][a-zA-Z0-9]+);")
_XML_BUILTIN_ENTITIES = frozenset({"amp", "lt", "gt", "quot", "apos"})


def _replace_named_entity(match: re.Match[str]) -> str:
    name = match.group(1)
    if name in _XML_BUILTIN_ENTITIES:
        return match.group(0)
    char = html.entities.html5.get(name + ";")
    if char is None:
        return match.group(0)
    return char


def _substitute_html_entities(text: str) -> str:
    return _NAMED_ENTITY_RE.sub(_replace_named_entity, text)


def _patched_fromstring(text):
    if isinstance(text, str):
        return _et_fromstring(_substitute_html_entities(text))
    if isinstance(text, (bytes, bytearray)):
        try:
            decoded = bytes(text).decode("utf-8")
        except UnicodeDecodeError:
            return _et_fromstring(text)
        return _et_fromstring(_substitute_html_entities(decoded).encode("utf-8"))
    return _et_fromstring(text)


_xml_like_mod.fromstring = _patched_fromstring


_PROTECTED_BLOCK_TAGS = frozenset({"pre", "script", "style"})
_PROTECTED_INLINE_TAGS = frozenset({"code", "kbd", "samp", "var"})
_BACKTICK_CODE_RE = re.compile(r"`[^`\n]+`")
_WHOLE_BACKTICK_CODE_RE = re.compile(r"^\s*`[^`\n]+`\s*$")


def _local_name(tag: str) -> str:
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[1].lower()
    return tag.lower()


def _is_protected_block(element: Element) -> bool:
    return _local_name(element.tag) in _PROTECTED_BLOCK_TAGS


def _is_protected_inline(element: Element) -> bool:
    return _local_name(element.tag) in _PROTECTED_INLINE_TAGS


def _has_protected_block_parent(segment: TextSegment) -> bool:
    return any(_is_protected_block(element) for element in segment.parent_stack)


def _has_protected_inline_parent(segment: TextSegment) -> bool:
    return any(_is_protected_inline(element) for element in segment.parent_stack)


def _has_protected_block_context(element: Element | None) -> bool:
    if element is None:
        return False
    return _is_protected_block(element)


class CodeProtection:
    """Protect code-like EPUB text from translation without relying on prompts."""

    def __init__(self) -> None:
        self._counter = 0
        self._placeholder_to_text: dict[str, str] = {}

    def protect_source_segments(self, segments):
        for segment in segments:
            if _has_protected_block_parent(segment):
                continue

            protected_text = self._protect_segment_text(segment)
            if protected_text is None:
                yield segment
                continue

            cloned = segment.clone()
            cloned.text = protected_text
            yield cloned

    def restore_translated_segments(self, segments):
        for segment in segments:
            restored_text = self._restore_placeholders(segment.text)
            if restored_text == segment.text:
                yield segment
                continue

            cloned = segment.clone()
            cloned.text = restored_text
            yield cloned

    def _protect_segment_text(self, segment: TextSegment) -> str | None:
        if _has_protected_inline_parent(segment) or _WHOLE_BACKTICK_CODE_RE.match(segment.text):
            return self._placeholder(segment.text)

        protected_text = _BACKTICK_CODE_RE.sub(lambda match: self._placeholder(match.group(0)), segment.text)
        if protected_text == segment.text:
            return None
        return protected_text

    def _placeholder(self, text: str) -> str:
        placeholder = f"__EN2KO_CODE_{self._counter:06d}__"
        self._counter += 1
        self._placeholder_to_text[placeholder] = text
        return placeholder

    def _restore_placeholders(self, text: str) -> str:
        for placeholder, original_text in self._placeholder_to_text.items():
            text = text.replace(placeholder, original_text)
        return text


_XMLInterrupter = _xml_interrupter_mod.XMLInterrupter
_orig_xml_interrupter_init = _XMLInterrupter.__init__
_orig_interrupt_source_text_segments = _XMLInterrupter.interrupt_source_text_segments
_orig_interrupt_translated_text_segments = _XMLInterrupter.interrupt_translated_text_segments


def _patched_xml_interrupter_init(self) -> None:
    _orig_xml_interrupter_init(self)
    self._en2ko_code_protection = CodeProtection()


def _patched_interrupt_source_text_segments(self, text_segments):
    protected = self._en2ko_code_protection.protect_source_segments(
        _orig_interrupt_source_text_segments(self, text_segments)
    )
    yield from protected


def _patched_interrupt_translated_text_segments(self, text_segments):
    restored = self._en2ko_code_protection.restore_translated_segments(
        _orig_interrupt_translated_text_segments(self, text_segments)
    )
    yield from restored


_XMLInterrupter.__init__ = _patched_xml_interrupter_init
_XMLInterrupter.interrupt_source_text_segments = _patched_interrupt_source_text_segments
_XMLInterrupter.interrupt_translated_text_segments = _patched_interrupt_translated_text_segments


_Submitter = _submitter_mod._Submitter
_orig_append_combined_after_tail = _Submitter._append_combined_after_tail
_TRANSLATION_STYLE = "background-color: rgba(255, 248, 190, 0.45);"


def _tag_like(reference: Element, local_name: str) -> str:
    if reference.tag.startswith("{"):
        namespace, _ = reference.tag[1:].split("}", 1)
        return f"{{{namespace}}}{local_name}"
    return local_name


def _wrap_inline_translation(reference: Element, combined: Element) -> tuple[Element, Element]:
    br = Element(_tag_like(reference, "br"))
    wrapper = Element(_tag_like(reference, "span"), {"style": _TRANSLATION_STYLE})
    wrapper.text = combined.text
    for child in list(combined):
        combined.remove(child)
        wrapper.append(child)
    return br, wrapper


def _insert_inline_translation(
    node_element: Element,
    insert_position: int,
    combined: Element,
) -> None:
    br, wrapper = _wrap_inline_translation(node_element, combined)
    node_element.insert(insert_position, br)
    node_element.insert(insert_position + 1, wrapper)


def _patched_append_combined_after_tail(
    self,
    node_element,
    text_segments,
    tail_element,
    anchor_element,
    append_to_end,
) -> None:
    if _has_protected_block_context(node_element):
        return

    if self._action != _SubmitKind.APPEND_TEXT:
        return _orig_append_combined_after_tail(
            self,
            node_element,
            text_segments,
            tail_element,
            anchor_element,
            append_to_end,
        )

    combined = self._combine_text_segments(text_segments)
    if combined is None:
        return

    if tail_element is not None:
        insert_position = _index_of_parent(node_element, tail_element) + 1
    elif append_to_end:
        insert_position = len(node_element)
    elif anchor_element is not None:
        ref_index = _index_of_parent(node_element, anchor_element)
        insert_position = ref_index if ref_index > 0 else 0
    else:
        insert_position = 0

    _insert_inline_translation(node_element, insert_position, combined)


_Submitter._append_combined_after_tail = _patched_append_combined_after_tail
