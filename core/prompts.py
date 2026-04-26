KOREAN_TRANSLATION_RULES = """\
- Translate the source English into natural, fluent Korean (자연스러운 한국어).
- Default narrative voice: 평서체(-다체). Keep tense and aspect faithful to the source.
- For dialogue, preserve register: maintain consistent speech levels (반말/존댓말) per speaker across the book.
- Personal names: do NOT translate. Transliterate to Hangul on first occurrence and place the original in parentheses, e.g. "존 스미스(John Smith)". Use the original spelling thereafter.
- Place names and brand names: prefer the established Korean spelling if widely known; otherwise transliterate with original in parentheses on first occurrence.
- Technical terms: translate, but on first occurrence add the original term in parentheses, e.g. "분산 시스템(distributed system)".
- Idioms and figures of speech: render the meaning naturally in Korean rather than literally; do not invent footnotes or explanations.
- Numbers, units, dates, and code remain unchanged unless idiomatic Korean usage requires conversion.
- Preserve the source's paragraph and sentence boundaries. Do not merge or split paragraphs.
- Do NOT omit, summarize, paraphrase loosely, or add any content not present in the source.
- Output Korean text only. Do not include English commentary, notes, or your own remarks.
"""
