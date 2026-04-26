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
import json
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from xml.etree.ElementTree import Element, fromstring as _et_fromstring

from epub_translator.epub import zip as _zipmod
from epub_translator.llm import Message, MessageRole
from epub_translator.segment import InlineSegment, TextSegment
from epub_translator.xml import xml_like as _xml_like_mod
from epub_translator.xml import index_of_parent as _index_of_parent
from epub_translator.translation import xml_interrupter as _xml_interrupter_mod
from epub_translator.xml_translator import translator as _translator_mod
from epub_translator.xml_translator import submitter as _submitter_mod
from epub_translator.xml_translator.callbacks import Callbacks
from epub_translator.xml_translator.stream_mapper import InlineSegmentMapping
from epub_translator.xml_translator.submitter import SubmitKind as _SubmitKind

_Zip = _zipmod.Zip
_orig_read = _Zip.read
_orig_replace = _Zip.replace
_orig_migrate = _Zip.migrate
_orig_list_files = _Zip.list_files


def _decode_path_text(path_text: str) -> str:
    return urllib.parse.unquote(path_text)


def _processed_key(path_text: str) -> str:
    decoded = _decode_path_text(path_text).replace("\\", "/")
    while decoded.startswith("./"):
        decoded = decoded[2:]
    return decoded


def _processed_path(path_text: str | Path) -> Path:
    if isinstance(path_text, Path):
        path_text = path_text.as_posix()
    return Path(_processed_key(path_text))


def _source_entry_name(self, path: Path) -> str:
    decoded_path = _decode_path_text(path.as_posix())
    names = self._source_zip.namelist()
    if decoded_path in names:
        return decoded_path

    dot_path = f"./{decoded_path}"
    if dot_path in names:
        return dot_path

    normalized = _processed_key(decoded_path)
    for name in names:
        if _processed_key(name) == normalized:
            return name

    return decoded_path


def _read(self, path: Path):
    return self._source_zip.open(_source_entry_name(self, path), "r")


def _replace(self, path: Path):
    self._processed_files.add(_processed_path(path))
    return self._target_zip.open(_decode_path_text(path.as_posix()), "w")


def _migrate(self, path: Path):
    source_name = _source_entry_name(self, path)
    source_info = self._source_zip.getinfo(source_name)
    with self._source_zip.open(source_name, "r") as source_file:
        content = source_file.read()
    self._target_zip.writestr(
        zinfo_or_arcname=source_info,
        data=content,
        compress_type=source_info.compress_type,
    )
    self._processed_files.add(_processed_path(source_name))


def _list_files(self, prefix_path: Path | None = None):
    all_files = self._source_zip.namelist()
    if prefix_path is None:
        return [_processed_path(file_path) for file_path in all_files]

    prefix = _processed_key(prefix_path.as_posix())
    if not prefix.endswith("/"):
        prefix += "/"
    return [
        _processed_path(file_path)
        for file_path in all_files
        if _processed_key(file_path).startswith(prefix)
    ]


def _exit(self, _exc_type, _exc_val, _exc_tb):
    try:
        if _exc_type is None:
            for file_path in self._source_zip.namelist():
                if file_path.endswith("/"):
                    continue
                processed_path = _processed_path(file_path)
                if processed_path not in self._processed_files:
                    self.migrate(Path(file_path))
    finally:
        self._target_zip.close()
        self._source_zip.close()

    return False


_Zip.read = _read
_Zip.replace = _replace
_Zip.migrate = _migrate
_Zip.list_files = _list_files
_Zip.__exit__ = _exit


_NAMED_ENTITY_RE = re.compile(r"&([a-zA-Z][a-zA-Z0-9]+);")
_ANGLE_CODE_TOKEN_RE = re.compile(r"</?[A-Z][A-Z0-9_.:-]*>")
_XML_BUILTIN_ENTITIES = frozenset({"amp", "lt", "gt", "quot", "apos"})
_HTML_BOOLEAN_ATTRIBUTES = frozenset(
    {
        "allowfullscreen",
        "async",
        "autofocus",
        "autoplay",
        "checked",
        "controls",
        "default",
        "defer",
        "disabled",
        "formnovalidate",
        "hidden",
        "inert",
        "ismap",
        "itemscope",
        "loop",
        "multiple",
        "muted",
        "nomodule",
        "novalidate",
        "open",
        "playsinline",
        "readonly",
        "required",
        "reversed",
        "selected",
    }
)


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


def _is_tag_start(text: str, index: int) -> bool:
    if index + 1 >= len(text):
        return False
    return text[index] == "<" and text[index + 1].isalpha()


def _find_opening_tag_end(text: str, start: int) -> int:
    quote: str | None = None
    index = start + 1
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == ">":
            return index
        index += 1
    return -1


def _read_attribute_value(tag_body: str, index: int) -> int:
    while index < len(tag_body) and tag_body[index].isspace():
        index += 1
    if index >= len(tag_body):
        return index
    if tag_body[index] in {"'", '"'}:
        quote = tag_body[index]
        index += 1
        while index < len(tag_body):
            if tag_body[index] == quote:
                return index + 1
            index += 1
        return index
    while index < len(tag_body) and not tag_body[index].isspace() and tag_body[index] not in {"/", ">"}:
        index += 1
    return index


def _normalize_boolean_attributes_in_tag(tag_body: str) -> str:
    match = re.match(r"[^\s/>]+", tag_body)
    if match is None:
        return tag_body

    parts = [tag_body[: match.end()]]
    index = match.end()

    while index < len(tag_body):
        char = tag_body[index]
        if char.isspace() or char == "/":
            parts.append(char)
            index += 1
            continue

        attr_start = index
        while index < len(tag_body) and not tag_body[index].isspace() and tag_body[index] not in {"=", "/", ">"}:
            index += 1
        attr_name = tag_body[attr_start:index]
        if not attr_name:
            parts.append(char)
            index += 1
            continue

        value_marker = index
        while value_marker < len(tag_body) and tag_body[value_marker].isspace():
            value_marker += 1

        if value_marker < len(tag_body) and tag_body[value_marker] == "=":
            value_end = _read_attribute_value(tag_body, value_marker + 1)
            parts.append(tag_body[attr_start:value_end])
            index = value_end
            continue

        if attr_name.lower() in _HTML_BOOLEAN_ATTRIBUTES:
            parts.append(f'{attr_name}="{attr_name.lower()}"')
        else:
            parts.append(attr_name)

    return "".join(parts)


def _normalize_html_boolean_attributes(text: str) -> str:
    normalized: list[str] = []
    index = 0

    while index < len(text):
        if not _is_tag_start(text, index):
            normalized.append(text[index])
            index += 1
            continue

        tag_end = _find_opening_tag_end(text, index)
        if tag_end == -1:
            normalized.append(text[index:])
            break

        tag_body = text[index + 1 : tag_end]
        normalized.append("<")
        normalized.append(_normalize_boolean_attributes_in_tag(tag_body))
        normalized.append(">")
        index = tag_end + 1

    return "".join(normalized)


def _prepare_xml_like_text(text: str) -> str:
    text = _substitute_html_entities(text)
    # 일부 EPUB/nav HTML에 삽입된 HTML boolean attribute를 XML 파서가
    # 원문 조각을 제거하지 않고 읽을 수 있게 만드는 보존형 정규화입니다.
    return _normalize_html_boolean_attributes(text)


def _patched_fromstring(text):
    if isinstance(text, str):
        return _et_fromstring(_prepare_xml_like_text(text))
    if isinstance(text, (bytes, bytearray)):
        try:
            decoded = bytes(text).decode("utf-8")
        except UnicodeDecodeError:
            return _et_fromstring(text)
        return _et_fromstring(_prepare_xml_like_text(decoded).encode("utf-8"))
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
        protected_text = _ANGLE_CODE_TOKEN_RE.sub(lambda match: self._placeholder(match.group(0)), protected_text)
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


_XMLTranslator = _translator_mod.XMLTranslator
_orig_translate_inline_segments = _XMLTranslator._translate_inline_segments
_PROTECTED_PLACEHOLDER_RE = re.compile(r"__EN2KO_CODE_\d{6}__")
_CODE_LIKE_RE = re.compile(
    r"(`[^`]+`|/[A-Za-z0-9_./{}:-]+|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[A-Z][A-Z0-9_]{2,})"
)
_PUNCT_OR_NUMBER_RE = re.compile(r"^[\s\d\W_]+$", re.UNICODE)
_LEAK_PATTERNS = (
    re.compile(r"please\s+provide", re.IGNORECASE),
    re.compile(r"\bnote\s*:", re.IGNORECASE),
    re.compile(r"죄송합니다"),
    re.compile(r"<\s*/?\s*rules\s*>", re.IGNORECASE),
    re.compile(r"translate\s+the\s+following", re.IGNORECASE),
    re.compile(r"korean\s+translation", re.IGNORECASE),
    re.compile(r"translated\s+text", re.IGNORECASE),
    re.compile(r"\bas\s+an\s+ai\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class _TranslationSlot:
    id: str
    index: int
    source_text: str


class _SlotValidationError(ValueError):
    pass


def _has_translatable_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    unprotected = _PROTECTED_PLACEHOLDER_RE.sub("", stripped)
    if not unprotected.strip():
        return False

    if not any(char.isalpha() for char in unprotected):
        return False

    if _PUNCT_OR_NUMBER_RE.match(unprotected):
        return False

    if _ANGLE_CODE_TOKEN_RE.fullmatch(unprotected.strip()):
        return False

    if _CODE_LIKE_RE.fullmatch(unprotected.strip()):
        return False

    return True


def _build_slot_context(source_segments: list[TextSegment]) -> str:
    return "".join(segment.text for segment in source_segments)


def _slot_system_prompt(base_rules: str, target_language: object, repair: bool = False) -> str:
    strictness = (
        "\nThe previous response was invalid. Return only valid JSON. "
        "Do not include notes, apologies, explanations, markdown, XML, or prompt text."
        if repair
        else ""
    )
    return f"""\
{base_rules}

Translate EPUB paragraph text slots into {target_language}.
Return JSON only with this exact shape:
{{"slots":[{{"id":"s0","text":"translated text"}}]}}

Rules:
- Use the paragraph context to translate short slots naturally.
- Translate only the listed slots. Keep every slot id exactly.
- Preserve placeholders such as __EN2KO_CODE_000001__ exactly if they appear.
- Preserve leading and trailing whitespace from each source slot.
- Keep code, API paths, model names, product names, numbers, and placeholders unchanged.
- Glossary: "bullshit" = "헛소리"; "On Bullshit" = "헛소리에 관하여"; "humbug" = "허풍"; "Black" as a person name = "블랙"; "short of lying" = "거짓말에는 못 미치는".
- Do not include "Please provide", "Note:", apologies, <rules>, XML, markdown, or explanations.{strictness}
"""


def _slot_user_prompt(context_text: str, slots: list[_TranslationSlot]) -> str:
    payload = {
        "context": context_text,
        "slots": [{"id": slot.id, "text": slot.source_text} for slot in slots],
    }
    return json.dumps(payload, ensure_ascii=False)


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise _SlotValidationError("response is not JSON") from None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise _SlotValidationError(f"response is not valid JSON: {exc}") from None

    if not isinstance(parsed, dict):
        raise _SlotValidationError("response root is not an object")
    return parsed


def _parse_slot_response(response: str, slots: list[_TranslationSlot]) -> dict[str, str]:
    parsed = _extract_json_object(response)
    raw_slots = parsed.get("slots", parsed.get("items"))
    if not isinstance(raw_slots, list):
        raise _SlotValidationError("response has no slots list")

    expected_ids = {slot.id for slot in slots}
    translations: dict[str, str] = {}
    for item in raw_slots:
        if not isinstance(item, dict):
            raise _SlotValidationError("slot item is not an object")
        slot_id = item.get("id")
        text = item.get("text")
        if not isinstance(slot_id, str) or not isinstance(text, str):
            raise _SlotValidationError("slot item must contain string id and text")
        if slot_id not in expected_ids:
            raise _SlotValidationError(f"unexpected slot id: {slot_id}")
        _validate_slot_translation(text)
        translations[slot_id] = text

    missing_ids = expected_ids - set(translations)
    if missing_ids:
        raise _SlotValidationError(f"missing slot ids: {sorted(missing_ids)}")
    return translations


def _validate_slot_translation(text: str) -> None:
    for pattern in _LEAK_PATTERNS:
        if pattern.search(text):
            raise _SlotValidationError(f"blocked leaked text: {pattern.pattern}")


def _apply_source_spacing(source: str, translated: str) -> str:
    leading = source[: len(source) - len(source.lstrip())]
    trailing = source[len(source.rstrip()) :]
    core = translated.strip()
    if not core:
        return source
    return f"{leading}{core}{trailing}"


def _request_slot_translations(
    self,
    context,
    context_text: str,
    slots: list[_TranslationSlot],
    repair: bool = False,
) -> dict[str, str]:
    response = context.request(
        input=[
            Message(
                role=MessageRole.SYSTEM,
                message=_slot_system_prompt(
                    base_rules=self._translation_llm.template("translate").render(
                        target_language=self._target_language,
                        user_prompt=self._user_prompt,
                    ),
                    target_language=self._target_language,
                    repair=repair,
                ),
            ),
            Message(role=MessageRole.USER, message=_slot_user_prompt(context_text, slots)),
        ],
        temperature=0.0 if repair else None,
    )
    translations = _parse_slot_response(response, slots)
    for slot in slots:
        _validate_slot_translation(translations[slot.id])
    return translations


def _deterministic_inline_segment_mapping(self, inline_segment: InlineSegment, context) -> InlineSegmentMapping | None:
    cloned = inline_segment.clone()
    source_segments = list(inline_segment)
    translated_segments = list(cloned)
    if len(source_segments) != len(translated_segments):
        return None

    slots = [
        _TranslationSlot(id=f"s{index}", index=index, source_text=source_segment.text)
        for index, source_segment in enumerate(source_segments)
        if _has_translatable_text(source_segment.text)
    ]

    changed = False
    if slots:
        control = getattr(self._translation_llm, "translation_control", None)
        if control is not None:
            control.checkpoint()

        context_text = _build_slot_context(source_segments)
        try:
            translations = _request_slot_translations(self, context, context_text, slots)
        except _SlotValidationError:
            try:
                translations = _request_slot_translations(self, context, context_text, slots, repair=True)
            except _SlotValidationError:
                translations = {}

        if control is not None:
            control.checkpoint()

        for slot in slots:
            translated_text = translations.get(slot.id)
            if translated_text is None:
                continue
            translated_text = _apply_source_spacing(slot.source_text, translated_text)
            translated_segments[slot.index].text = translated_text
            changed = changed or translated_text != slot.source_text

    if not changed:
        return None

    text_segments = list(cloned)
    if not text_segments:
        return None
    return inline_segment.parent, text_segments


def _patched_translate_inline_segments(
    self,
    inline_segments: list[InlineSegment],
    callbacks: Callbacks,
) -> list[InlineSegmentMapping | None]:
    control = getattr(self._translation_llm, "translation_control", None)
    if control is not None:
        control.checkpoint()

    if not getattr(self._translation_llm, "deterministic_xml_fill", False):
        mappings = _orig_translate_inline_segments(
            self,
            inline_segments=inline_segments,
            callbacks=callbacks,
        )
        if control is not None:
            control.checkpoint()
        return mappings

    with self._translation_llm.context(cache_seed_content=self._cache_seed_content) as context:
        return [
            _deterministic_inline_segment_mapping(self, inline_segment, context)
            for inline_segment in inline_segments
        ]


_XMLTranslator._translate_inline_segments = _patched_translate_inline_segments
