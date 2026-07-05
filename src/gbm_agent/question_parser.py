from __future__ import annotations
import json
from typing import Any, Dict, Optional
from .config import GPT_MODEL, client
from .prompts import QUESTION_PARSER_SYSTEM_PROMPT, question_parser_user_prompt


class QuestionParser:
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or GPT_MODEL

    def parse(self, question: str) -> Dict[str, Any]:
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUESTION_PARSER_SYSTEM_PROMPT},
                    {"role": "user", "content": question_parser_user_prompt(question)},
                ],
                temperature=0.1,
            )
            content = resp.choices[0].message.content or "{}"
            content = content.replace("```json", "").replace("```", "").strip()
            obj = json.loads(content)
            if not isinstance(obj, dict):
                raise ValueError("parser output is not a dict")
        except Exception as exc:
            obj = {
                "type": "other",
                "population": None,
                "intervention": None,
                "comparator": None,
                "outcome": None,
                "time_focus": "any",
                "extras": [],
                "parser_error": str(exc),
            }
        obj.setdefault("raw_question", question)
        return obj


def parse_question(question: str) -> Dict[str, Any]:
    parser = QuestionParser()
    return parser.parse(question)
