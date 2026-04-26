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


# 원문 영어를 자연스럽고 유창한 한국어(자연스러운 한국어)로 번역한다.
# 기본 서술체: 평서체(-다체). 시제와 상(aspect)은 원문에 충실하게 유지한다.
# 대화문의 경우 어투를 보존한다: 각 화자의 말투(반말/존댓말)를 책 전체에 걸쳐 일관되게 유지한다.
# 인명: 번역하지 않는다. 첫 등장 시 한글로 음차 표기하고 원어를 괄호 안에 병기한다. 예) "존 스미스(John Smith)". 이후에는 원어 표기를 사용한다.
# 지명 및 브랜드명: 널리 알려진 한국어 표기가 있으면 이를 우선 사용하고, 그렇지 않으면 첫 등장 시 음차 표기 후 원어를 괄호 안에 병기한다.
# 전문 용어: 번역하되, 첫 등장 시 원어를 괄호 안에 병기한다. 예) "분산 시스템(distributed system)".
# 관용구 및 비유적 표현: 직역하지 않고 의미를 자연스러운 한국어로 전달한다. 각주나 부연 설명을 임의로 추가하지 않는다.
# 숫자, 단위, 날짜, 코드는 한국어 관용 표현으로의 변환이 필요한 경우를 제외하고 그대로 유지한다.
# 원문의 문단 및 문장 경계를 보존한다. 문단을 합치거나 분리하지 않는다.
# 원문의 내용을 생략, 요약, 느슨한 의역하거나, 원문에 없는 내용을 추가하지 않는다.
# 한국어 번역문만 출력한다. 영어 해설, 주석, 또는 번역자의 코멘트를 포함하지 않는다.
