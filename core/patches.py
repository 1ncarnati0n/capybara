"""
Workarounds for upstream bugs in oomol-lab/epub-translator.

1. IRI-encoded hrefs vs. raw ZIP entries
   The library reads chapter / TOC / manifest hrefs straight out of the OPF
   (see `epub_translator/epub/spines.py` and `toc.py`) and then passes them to
   `zipfile.ZipFile.open(...)`. EPUB spec requires those hrefs to be
   IRI-encoded (spaces → %20, parentheses → %28/%29, etc.), but ZIP entries
   store the raw filename. We monkey-patch `epub_translator.epub.zip.Zip` to
   URL-decode every path before talking to `zipfile`.

2. HTML named entities in XHTML chapters
   `xml.etree.ElementTree.fromstring` only recognises the five XML built-in
   entities. XHTML chapters routinely contain `&nbsp;`, `&copy;`, `&mdash;`,
   etc., which are declared in the XHTML DTD that ElementTree never fetches.
   We monkey-patch `epub_translator.xml.xml_like.fromstring` to replace HTML5
   named entities with their Unicode characters before parsing.

3. Inline append presentation
   In APPEND_TEXT mode, upstream appends the translation to the same text node
   with a single space. For Korean review EPUBs we want the translated text on
   the next line with a subtle highlight, while keeping REPLACE untouched.

Importing this module applies these patches as a side effect.
"""

from __future__ import annotations

import html.entities
import re
import urllib.parse
from pathlib import Path
from xml.etree.ElementTree import Element, fromstring as _et_fromstring

from epub_translator.epub import zip as _zipmod
from epub_translator.xml import xml_like as _xml_like_mod
from epub_translator.xml import index_of_parent as _index_of_parent
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
