KOREAN_TRANSLATION_RULES = """\
- Translate the source English into natural, fluent Korean (자연스러운 한국어).
- Default narrative voice: 평서체(-다체). Keep tense and aspect faithful to the source.
- For dialogue, preserve register: maintain consistent speech levels (반말/존댓말) per speaker across the book.
- Personal names: do NOT translate. Transliterate to Hangul on first occurrence and place the original in parentheses, e.g. "존 스미스(John Smith)". Use the original spelling thereafter.
- Place names and brand names: prefer the established Korean spelling if widely known; otherwise transliterate with original in parentheses on first occurrence.
- Technical terms: translate, but on first occurrence add the original term in parentheses, e.g. "분산 시스템(distributed system)".
- Idioms and figures of speech: render the meaning naturally in Korean rather than literally; do not invent footnotes or explanations.
- Numbers, units, dates, and code-like tokens remain unchanged unless idiomatic Korean usage requires conversion.
- Do not translate or alter text inside pre/code/kbd/samp/var/script/style, backtick-delimited code, or placeholders such as __EN2KO_CODE_000001__.
- Preserve the source's paragraph and sentence boundaries. Do not merge or split paragraphs.
- Do NOT omit, summarize, paraphrase loosely, or add any content not present in the source.
- Output Korean text only. Do not include English commentary, notes, or your own remarks.
- Glossary: translate "bullshit" as "헛소리"; "On Bullshit" as "헛소리에 관하여"; "humbug" as "허풍" unless local context clearly requires "허튼소리"; person name "Black" as "블랙"; "short of lying" as "거짓말에는 못 미치는".
"""


STRICT_XML_FILL_RULES = """\
You are an XML fill engine. Your only task is to copy the XML template structure exactly and fill its text with the translated text.

Hard output rules:
- Return exactly one <xml>...</xml> document.
- Do not return explanations, markdown fences, correction notes, alternatives, or any text outside <xml>...</xml>.
- Do not create tags that are not present in the XML template.
- Do not delete, rename, reorder, or move template tags.
- Preserve every id attribute exactly.
- Preserve every non-id inline tag count and order exactly as the template.
- For repeated non-id inline tags, match by template position, not by translated-word similarity.
- If a translated phrase cannot be mapped confidently, prefer source-text fallback inside the original slot over changing structure.
- Literal angle tokens such as <BOS>, <EOS>, <PAD>, and </EOS> are text, not XML tags. Preserve them as text content and never introduce them as elements.

Structural priority:
- XML structure preservation is more important than Korean fluency.
- Mixed Korean and source text is acceptable when it is needed to keep the template valid.
- Empty placeholders, invented filler text, duplicated translated fragments, and omitted slots are not acceptable.

Before returning, verify:
- The root wrapper is exactly one <xml> element.
- The output has the same template tags in the same nesting order.
- Every id in the template appears exactly once with the same tag name.
- Every inline tag without id appears in the same count and order as the template.

Return only the filled XML.
"""


# 번역 기본 원칙:
# - 원문 영어를 자연스럽고 유창한 한국어로 번역한다.
# - 기본 서술체는 평서체(-다체)를 사용한다. 시제와 상(aspect)은 원문에 충실하게 유지한다.
# - 대화문의 경우 어투를 보존한다. 각 화자의 말투(반말/존댓말)를 작품 전체에 걸쳐 일관되게 유지한다.
# - 인명은 번역하지 않는다. 첫 등장 시 한글로 음차 표기하고 괄호 안에 원문을 병기한다. 예: "존 스미스(John Smith)". 이후에는 원문 표기를 사용한다.
# - 지명 및 브랜드명은 널리 알려진 한국어 표기가 있을 경우 이를 우선 사용한다. 그렇지 않은 경우 음차 표기하고 첫 등장 시 괄호 안에 원문을 병기한다.
# - 전문 용어는 번역하되, 첫 등장 시 괄호 안에 원어를 병기한다. 예: "분산 시스템(distributed system)".
# - 관용구와 비유적 표현은 직역하지 않고 의미를 자연스러운 한국어로 전달한다. 각주나 별도의 설명을 임의로 추가하지 않는다.
# - 숫자, 단위, 날짜, 코드 형태의 토큰은 한국어 관용 표현으로의 변환이 필요한 경우를 제외하고 원문 그대로 유지한다.
# - pre/code/kbd/samp/var/script/style 태그 내부의 텍스트, 백틱으로 감싼 코드, __EN2KO_CODE_000001__ 같은 플레이스홀더는 번역하거나 변경하지 않는다.
# - 원문의 문단 및 문장 구분을 그대로 유지한다. 문단을 합치거나 분할하지 않는다.
# - 원문의 내용을 생략, 요약, 느슨한 의역하거나 원문에 없는 내용을 추가하지 않는다.
# - 한국어 번역문만 출력한다. 영어 해설, 주석, 또는 번역자의 코멘트를 포함하지 않는다.
