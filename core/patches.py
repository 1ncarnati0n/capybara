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

Importing this module applies both patches as a side effect.
"""

from __future__ import annotations

import html.entities
import re
import urllib.parse
from pathlib import Path
from xml.etree.ElementTree import fromstring as _et_fromstring

from epub_translator.epub import zip as _zipmod
from epub_translator.xml import xml_like as _xml_like_mod

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
