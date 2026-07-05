from __future__ import annotations
import json
import sys
import time

try:
    import io

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
import re
import os
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
from openai import BadRequestError
from openai import APITimeoutError, RateLimitError
from .config import (
    client,
    GPT_MODEL,
    LLM_BACKEND,
    GEMINI_ENABLED,
    gemini_generate_text,
)
from .question_parser import parse_question
from .prompts import (
    FINAL_SECTION_TITLES,
    PATIENT_FRIENDLY_FALLBACK,
    SPECIALIST_DISCUSSION_PROMPT,
    SPECIALIST_INITIAL_ISOLATION_PROMPT,
    SPECIALIST_ROLE_PROMPTS,
    pipeline_system_prompt,
    render_ultra_prompt,
    specialist_system_prompt,
)
from .advanced_tools import (
    llm_mri,
    llm_gigatime,
    search_local_guidelines,
    search_local_facts,
    search_google,
    search_pubmed,
    read_webpage,
    gigatime_infer,
    roam_infer,
    raspr_run,
    scrna_pipeline_run,
    TOOLS_SCHEMA,
)

ToolResult = Dict[str, Any]
Citation = Dict[str, Any]
MAX_AUDIT_RETRIES = 2
DISCUSSION_ROUNDS = 2
MAX_REACT_STEPS = 15
MEMORY_DIR = Path("patient_memory")
WATCH_QUEUE_FILE = Path("run_logs") / "watch_queue.json"
UNCERTAINTY_HIGH_THRESHOLD = 0.75
UNCERTAINTY_LOW_THRESHOLD = 0.40
GRADE_LEVELS = {
    "A": "Status update.",
    "B": "Status update.",
    "C": "Status update.",
    "D": "Status update.",
    "I": "Status update.",
}


def _now_hhmm(tz_name: str = "Asia/Singapore") -> str:
    try:
        return datetime.now(ZoneInfo(tz_name)).strftime("%H:%M")
    except Exception:
        return datetime.now().strftime("%H:%M")


def _normalize_scrna_cluster_label(x: str) -> str:
    s = (x or "").strip()
    if not s:
        return ""
    sl = s.lower().strip()
    m = re.search(r"(cluster\s*([0-3]))", sl)
    if m:
        return f"cluster{m .group (2 )}"
    m2 = re.search(r"\b([0-3])\b", sl)
    if m2 and sl in ["0", "1", "2", "3"]:
        return f"cluster{m2 .group (1 )}"
    return s


def _extract_numbered_questions(text: str) -> List[str]:
    if not text:
        return []
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(r"(:m)^\s*(\d{1,2})\s*(::[\. \)]| )\s*")
    matches = list(pattern.finditer(s))
    if not matches:
        return []
    items: List[str] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if (idx + 1) < len(matches) else len(s)
        chunk = s[start:end].strip()
        if len(chunk) >= 3:
            items.append(chunk)
    return items


def _compact_one_line_summary(summary: str, max_len: int = 220) -> str:
    s = (summary or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s*\n\s*", " | ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


@dataclass
class Plan:
    parsed: Dict[str, Any]
    suggested_tools: List[str]
    notes: str


@dataclass
class AgentMessage:
    agent_name: str
    role: str
    content: str
    citations: List[Citation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedBus:
    messages: List[AgentMessage] = field(default_factory=list)

    def post(self, msg: AgentMessage) -> None:
        self.messages.append(msg)

    def get_thread_text(self, exclude_agent: Optional[str] = None) -> str:
        lines: List[str] = []
        for m in self.messages:
            if exclude_agent and m.agent_name == exclude_agent:
                continue
            lines.append(f" {m .agent_name } {m .content }")
        return "\n\n".join(lines)

    def get_full_thread_text(self) -> str:
        lines: List[str] = []
        for m in self.messages:
            lines.append(f" {m .agent_name } {m .content }")
        return "\n\n".join(lines)


class LLMCallerMixin:
    model: str
    _trace_logs: List[str]
    _step_counter: int
    _stream_callback: Any

    def _sanitize_str(self, s: Any) -> str:
        if s is None:
            return ""
        t = str(s)
        if len(t) > 12000:
            t = t[:12000]
        cleaned = []
        for ch in t:
            cp = ord(ch)
            if cp == 0 or cp == 11 or cp == 12:
                continue
            if ch >= " " or ch in ("\n", "\r", "\t"):
                cleaned.append(ch)
        return "".join(cleaned)

    def _model_is_gpt5_family(self, model: str) -> bool:
        return (model or "").lower().strip().startswith("gpt-5")

    def _emit_stream(self, line: str) -> None:
        try:
            cb = getattr(self, "_stream_callback", None)
            if cb:
                cb(line)
        except Exception:
            pass

    def _log_ts(self, message: str) -> None:
        import sys as _sys

        line = f"[{_now_hhmm ()}] {message }"
        print(line, flush=True)
        _sys.stdout.flush()
        self._trace_logs.append(line)
        self._emit_stream(line)

    def _log(self, message: str) -> None:
        line = f"{message }"
        print(line, flush=True)
        self._trace_logs.append(line)
        self._emit_stream(line)

    def _log_step(self, title: str) -> None:
        self._step_counter += 1
        import sys as _sys

        _sys.stdout.flush()
        _sys.stderr.flush()
        self._log_ts(f"Step {self ._step_counter }: {title }")

    def _chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout_override: Optional[float] = None,
    ):
        _caller = self.__class__.__name__
        model = model_name or self.model
        if LLM_BACKEND == "gemini_native" and model.lower().startswith("gemini"):
            return self._chat_completion_gemini(
                messages=messages,
                model_name=model,
                temperature=temperature,
                tools=tools,
                max_tokens=max_tokens,
                timeout_override=timeout_override,
            )
        if tools:
            print(f"  [{_caller }] message LLM message message ...", flush=True)
        else:
            print(f"  [{_caller }] message LLM message {len (messages )} message ...", flush=True)
        safe_messages: List[Dict[str, Any]] = []
        for m in messages:
            if isinstance(m, dict):
                msg = m.copy()
            elif hasattr(m, "model_dump"):
                msg = m.model_dump()
            else:
                msg = {"role": "user", "content": str(m)}
            if "content" not in msg or msg["content"] is None:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    pass
                else:
                    msg["content"] = ""
            if msg.get("role") == "tool" and not isinstance(msg.get("content"), str):
                msg["content"] = str(msg.get("content"))
            if isinstance(msg.get("content"), str):
                msg["content"] = self._sanitize_str(msg["content"])
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"] or []:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        if isinstance(fn.get("arguments"), dict):
                            fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
                        elif fn.get("arguments") is None:
                            fn["arguments"] = "{}"
            safe_messages.append(msg)
        kwargs: Dict[str, Any] = dict(model=model, messages=safe_messages)
        if not self._model_is_gpt5_family(model):
            kwargs["temperature"] = temperature
        else:
            if float(temperature) == 1.0:
                kwargs["temperature"] = 1.0
        if tools is not None:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        max_retries = 4
        backoff_s = 1.0
        last_err: Optional[Exception] = None
        _caller_name = self.__class__.__name__
        _timeout_map = {"SynthesizerAgent": 600.0}
        kwargs["timeout"] = (
            timeout_override
            if timeout_override is not None
            else _timeout_map.get(_caller_name, 600.0)
        )
        print(f"  [{_caller_name }] timeout={kwargs ['timeout']}s", flush=True)
        for attempt in range(1, max_retries + 1):
            try:
                return client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                msg_str = str(e)
                if ("temperature" in msg_str) and (
                    "Only the default (1) value is supported" in msg_str
                    or "unsupported_value" in msg_str
                ):
                    self._log("Status update.")
                    kwargs.pop("temperature", None)
                    try:
                        return client.chat.completions.create(**kwargs)
                    except Exception as e2:
                        last_err = e2
                        break
                self._log(f" OpenAI BadRequestError: {e }")
                raise
            except RateLimitError as e:
                last_err = e
                err_str = str(e)
                if "insufficient_quota" in err_str or "exceeded your current quota" in err_str:
                    print(f"\n [OpenAI] API message message message {err_str [:200 ]}", flush=True)
                    self._log(f" OpenAI message {err_str }")
                    raise
                self._log(f" OpenAI RateLimitError (attempt {attempt }/{max_retries }): {e }")
                if attempt < max_retries:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 1.8, 8.0)
                    continue
                raise
            except APITimeoutError as e:
                last_err = e
                self._log(f" OpenAI APITimeoutError (attempt {attempt }/{max_retries }): {e }")
                print(
                    f" [OpenAI] message >{kwargs .get ('timeout',120 )}s message {attempt }/{max_retries } ...",
                    flush=True,
                )
                if attempt < max_retries:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 1.8, 8.0)
                    continue
                raise
            except Exception as e:
                last_err = e
                self._log(f" OpenAI transient error (attempt {attempt }/{max_retries }): {e }")
                if attempt < max_retries:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 1.8, 8.0)
                    continue
                raise
        if last_err:
            raise last_err
        raise RuntimeError("Unknown error in _chat_completion")

    def _chat_completion_gemini(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        temperature: float = 0.2,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        timeout_override: Optional[float] = None,
    ):
        import google.generativeai as genai
        from google.generativeai import types as genai_types

        _caller = self.__class__.__name__
        print(f"  [{_caller }] message Gemini ({model_name }) message...", flush=True)
        system_parts = []
        history = []
        for m in messages:
            role = m.get("role", "user")
            raw_content = m.get("content") or ""
            if isinstance(raw_content, list):
                text_content = " ".join(
                    p.get("text", "") for p in raw_content if isinstance(p, dict)
                )
            else:
                text_content = str(raw_content)
            text_content = self._sanitize_str(text_content)
            if role == "system":
                system_parts.append(text_content)
            elif role == "user":
                history.append({"role": "user", "parts": [text_content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [text_content]})
            elif role == "tool":
                history.append({"role": "user", "parts": [f"[message] {text_content }"]})
        system_instruction = "\n\n".join(system_parts) if system_parts else None
        gen_config = genai_types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 32000,
        )
        model_kwargs: Dict[str, Any] = {"generation_config": gen_config}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction
        gemini_tools = None
        if tools:
            try:
                func_decls = []
                for t in tools:
                    fn = t.get("function") or {}
                    params = fn.get("parameters") or {}
                    func_decls.append(
                        genai_types.FunctionDeclaration(
                            name=fn.get("name", ""),
                            description=fn.get("description", ""),
                            parameters=params,
                        )
                    )
                gemini_tools = [genai_types.Tool(function_declarations=func_decls)]
                model_kwargs["tools"] = gemini_tools
            except Exception as _te:
                self._log(f"[Gemini] message {_te } message")
        gemini_model = genai.GenerativeModel(model_name=model_name, **model_kwargs)
        _timeout = timeout_override or 600.0
        max_retries = 3
        backoff_s = 2.0
        last_err: Optional[Exception] = None
        print(f"  [{_caller }] Gemini timeout={_timeout }s", flush=True)
        for attempt in range(1, max_retries + 1):
            try:
                if history:
                    last_turn = history[-1]
                    prior_history = history[:-1]
                    chat = gemini_model.start_chat(history=prior_history)
                    gemini_resp = chat.send_message(
                        last_turn["parts"][0],
                        request_options={"timeout": _timeout},
                    )
                else:
                    gemini_resp = gemini_model.generate_content(
                        "Status update.",
                        request_options={"timeout": _timeout},
                    )
                break
            except Exception as e:
                last_err = e
                self._log(f" Gemini error (attempt {attempt }/{max_retries }): {e }")
                print(
                    f" [Gemini] message {type (e ).__name__ } message {attempt }/{max_retries } ...",
                    flush=True,
                )
                if attempt < max_retries:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, 10.0)
                    continue
                raise
        else:
            if last_err:
                raise last_err
        return _GeminiResponseWrapper(gemini_resp)


class _GeminiToolCall:
    def __init__(self, name: str, args: dict, call_id: str):
        self.id = call_id
        self.type = "function"
        self.function = type(
            "_Fn",
            (),
            {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        )()


class _GeminiMessage:
    def __init__(self, text: str, tool_calls=None):
        self.content = text
        self.tool_calls = tool_calls or []
        self.role = "assistant"

    def model_dump(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": (
                [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in self.tool_calls
                ]
                if self.tool_calls
                else None
            ),
        }


class _GeminiChoice:
    def __init__(self, message: _GeminiMessage):
        self.message = message
        self.finish_reason = "stop"


class _GeminiResponseWrapper:
    def __init__(self, gemini_resp):
        self._raw = gemini_resp
        self.choices = [self._build_choice(gemini_resp)]

    @staticmethod
    def _build_choice(resp) -> _GeminiChoice:
        tool_calls = []
        try:
            for part in resp.candidates[0].content.parts or []:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    call_id = f"gemini_call_{fc .name }_{id (fc )}"
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(_GeminiToolCall(fc.name, args, call_id))
        except Exception:
            pass
        text = ""
        try:
            text = resp.text or ""
        except Exception:
            try:
                parts = resp.candidates[0].content.parts
                text = "".join(p.text for p in parts if hasattr(p, "text") and p.text)
            except Exception:
                text = ""
        return _GeminiChoice(_GeminiMessage(text=text, tool_calls=tool_calls or None))


class PipelineAgent(LLMCallerMixin):
    def __init__(
        self,
        model: str,
        tools_schema: List[Dict[str, Any]],
        trace_logs: List[str],
        step_counter_ref: List[int],
        stream_callback: Any = None,
    ) -> None:
        self.model = model
        self.tools_schema = tools_schema
        self._trace_logs = trace_logs
        self._step_counter_ref = step_counter_ref
        self._stream_callback = stream_callback
        self._tool_counts: Dict[str, int] = {}
        self._tool_events: List[Dict[str, Any]] = []
        self._tool_last_summary: Dict[str, str] = {}
        self._tool_desc: Dict[str, str] = {}
        for t in self.tools_schema:
            fn = t.get("function") or {}
            name = fn.get("name")
            desc = fn.get("description") or ""
            if name:
                self._tool_desc[name] = desc

    @property
    def _step_counter(self) -> int:
        return self._step_counter_ref[0]

    @_step_counter.setter
    def _step_counter(self, v: int) -> None:
        self._step_counter_ref[0] = v

    def _get_default_vision_model(self) -> str:
        vm = (os.environ.get("VISION_MODEL") or "").strip()
        return vm or getattr(self, "model", None) or GPT_MODEL

    def _call_llm_mri_compat(
        self,
        input_path: str,
        images: Optional[List[str]] = None,
        question: str = "",
        max_images: int = 12,
        model: Optional[str] = None,
    ) -> ToolResult:
        try:
            imgs: List[str] = []
            if isinstance(images, list) and images:
                imgs = [str(x) for x in images if str(x).strip()]
            elif input_path:
                imgs = [str(input_path)]
            vision_model = (model or "").strip() or self._get_default_vision_model()
            try:
                return llm_mri(
                    input_path=str(input_path),
                    question=str(question or ""),
                    max_images=int(max_images or 12),
                    model=vision_model,
                )
            except TypeError as e:
                if ("unexpected keyword" not in str(e)) and (
                    "got an unexpected keyword" not in str(e)
                ):
                    raise
            try:
                return llm_mri(images=imgs, question=str(question or ""), model=vision_model)
            except TypeError:
                pass
            try:
                return llm_mri(images=imgs, model=vision_model)
            except Exception as e:
                return {
                    "ok": False,
                    "tool_name": "llm_mri",
                    "records": [],
                    "formatted": f"llm_mri Error: {e }",
                    "error": f"llm_mri Error: {e }",
                }
        except Exception as e:
            return {
                "ok": False,
                "tool_name": "llm_mri",
                "records": [],
                "formatted": f"llm_mri Error: {e }",
                "error": f"llm_mri Error: {e }",
            }

    def _derive_reason_fallback(
        self, tool_name: str, args: Dict[str, Any], plan: Optional[Plan]
    ) -> str:
        desc = self._tool_desc.get(tool_name, "")
        qtype = (plan.parsed.get("type") if plan else "") or ""
        reasons = {
            "search_local_guidelines": f"message={qtype } message {desc }",
            "search_local_facts": "Status update.",
            "search_pubmed": "Status update.",
            "search_google": "Status update.",
            "read_webpage": "Status update.",
            "llm_mri": "Status update.",
            "gigatime_infer": "Status update.",
            "llm_gigatime": "Status update.",
            "roam_infer": "Status update.",
            "raspr_run": "Status update.",
            "scrna_pipeline_run": "Status update.",
        }
        return reasons.get(tool_name, desc or "Status update.")

    def _summarize_tool_result(self, tool_name: str, tool_result: ToolResult, topn: int = 5) -> str:
        recs = tool_result.get("records") or []
        if not recs:
            return "Status update."
        lines: List[str] = []
        if tool_name == "search_local_guidelines":
            lines.append("Status update.")
            for i, r in enumerate(recs[:topn], start=1):
                title = r.get("title") or "Status update."
                sc = r.get("score_final")
                sc_str = f"{sc :.2f}" if isinstance(sc, (int, float)) else "N/A"
                lines.append(f" {i }. {title } | final_score={sc_str }")
        elif tool_name == "search_local_facts":
            lines.append("Status update.")
            for i, r in enumerate(recs[:topn], start=1):
                title = r.get("title") or "Status update."
                hr = r.get("hazard_ratio")
                outcome = r.get("outcome") or ""
                lines.append(f" {i }. {title } | {outcome } | HR={hr }")
        elif tool_name == "search_pubmed":
            lines.append("Status update.")
            for i, r in enumerate(recs[:topn], start=1):
                pmid = r.get("pmid") or "N/A"
                title = r.get("title") or "Status update."
                lines.append(f" {i }. PMID {pmid } | {title }")
        elif tool_name == "search_google":
            lines.append("Status update.")
            for i, r in enumerate(recs[:topn], start=1):
                title = r.get("title") or "Status update."
                lines.append(f" {i }. {title }")
        elif tool_name == "llm_mri":
            formatted = (tool_result.get("formatted") or "").strip()
            brief = "\n".join(formatted.splitlines()[:18])
            return "Status update." + brief
        elif tool_name == "gigatime_infer":
            r0 = recs[0] if recs else {}
            imgs = r0.get("generated_images") or []
            maps = r0.get("map_images") or []
            llm_block = r0.get("llm_gigatime") or {}
            lines.append("Status update.")
            lines.append(f" - message: {len (imgs )}")
            lines.append(f" - Map_*.png message: {len (maps )}")
            if llm_block.get("ok"):
                llm_recs = llm_block.get("records") or []
                if llm_recs:
                    analysis = (llm_recs[0] or {}).get("analysis") or {}
                    overview = analysis.get("overview") or ""
                    if overview:
                        lines.append(f" - LLMmessage: {str (overview )[:200 ]}")
            elif llm_block.get("error"):
                lines.append(f" - LLMmessage: {llm_block .get ('error')}")
        elif tool_name == "llm_gigatime":
            r0 = recs[0] if recs else {}
            analysis = r0.get("analysis") or {}
            lines.append("Status update.")
            lines.append(f" - visualizations_dir: {r0 .get ('visualizations_dir')}")
            lines.append(f" - Map message: {len (r0 .get ('map_paths')or [])}")
            overview = analysis.get("overview") or ""
            if overview:
                lines.append(f" - overview: {str (overview )[:200 ]}")
            notable = analysis.get("notable_markers") or []
            if notable:
                lines.append(f" - notable_markersmessage: {len (notable )}")
        elif tool_name == "roam_infer":
            r0 = recs[0] if recs else {}
            pred = r0.get("predicted_subtype_guess")
            rc = r0.get("return_code")
            lines.append("Status update.")
            lines.append(f" - predicted_subtype_guess: {pred }")
            lines.append(f" - return_code: {rc }")
        elif tool_name == "raspr_run":
            r0 = recs[0] if recs else {}
            pred = r0.get("survival_prediction") or r0.get("predicted_survival") or {}
            lines.append("Status update.")
            lines.append(f" - message: {pred }")
        elif tool_name == "scrna_pipeline_run":
            r0 = recs[0] if recs else {}
            lines.append("Status update.")
            lines.append(f" - outdir: {r0 .get ('outdir')}")
            lines.append(f" - preannotation_h5ad: {r0 .get ('preannotation_h5ad')}")
            lines.append(f" - final_h5ad: {r0 .get ('final_h5ad')}")
            pred_guess = r0.get("prediction_guess")
            predicted_cluster = r0.get("predicted_cluster")
            effective = predicted_cluster or pred_guess
            lines.append(f" - predicted_cluster(CSV): {predicted_cluster }")
            lines.append(f" - prediction_guess(log): {pred_guess }")
            lines.append(f" - effective_cluster: {effective }")
            if effective:
                cl = _normalize_scrna_cluster_label(str(effective))
                interp = (
                    _SCRNA_CLUSTER_INTERP.get(cl) or _SCRNA_CLUSTER_INTERP.get(str(effective)) or ""
                )
                if interp:
                    lines.append(" - subtype_interpretation(prior): " + interp)
                if cl == "cluster3":
                    lines.append(" - drug_sensitivity_hint(prior): " + _SCRNA_DRUG_HINT_CLUSTER3)
                else:
                    lines.append("Status update.")
            else:
                lines.append("Status update.")
                lines.append("Status update.")
            lines.append("Status update.")
        else:
            return f"message message {len (recs )} message "
        return "\n".join(lines)

    def _call_tool(
        self, name: str, arguments: Dict[str, Any], plan: Optional[Plan] = None
    ) -> ToolResult:
        reason = (arguments.get("reason") or "").strip()
        args_clean = {k: v for k, v in arguments.items() if k != "reason"}
        if not reason:
            reason = self._derive_reason_fallback(name, args_clean, plan)
        print(f"\n [PipelineAgent] Calling Tool: {name }", flush=True)
        print(f" - Reason: {reason }", flush=True)
        print(f" - Params: {json .dumps (args_clean ,ensure_ascii =False )}", flush=True)
        self._tool_counts[name] = self._tool_counts.get(name, 0) + 1
        _tool_startup_msgs = {
            "raspr_run": ["Status update.", "Status update."],
            "gigatime_infer": ["Status update.", "Status update.", "Status update."],
            "llm_gigatime": ["Status update."],
            "llm_mri": ["Status update."],
            "roam_infer": ["Status update.", "Status update.", "Status update."],
            "scrna_pipeline_run": ["Status update.", "Status update."],
        }
        for msg in _tool_startup_msgs.get(name, []):
            print(msg, flush=True)
        t0 = time.perf_counter()
        self._log(f"message {name }")
        self._log(f" - message {reason }")
        _slow_tools = {"scrna_pipeline_run", "raspr_run", "gigatime_infer", "roam_infer"}
        if name in _slow_tools:
            import datetime

            _start_ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n{'='*60 }", flush=True)
            print(f" [{_start_ts }] message {name }", flush=True)
            print(f" message message...", flush=True)
            print(f" message message", flush=True)
            print(f"{'='*60 }\n", flush=True)
        out = self._dispatch_tool(name, args_clean)
        dt = time.perf_counter() - t0
        rec_count = len(out.get("records", []) or [])
        _status = " OK" if out.get("ok") else " Error"
        print(f"\n{' '*60 }", flush=True)
        print(
            f"{_status } [{name }] message | message {dt :.1f}s | message {rec_count } message",
            flush=True,
        )
        print(f"{' '*60 }", flush=True)
        summary_text = self._summarize_tool_result(name, out)
        summary_one_line = _compact_one_line_summary(summary_text)
        self._tool_last_summary[name] = summary_one_line
        self._log(
            f"message {name } | message {dt :.2f}s | message {rec_count } message | ok={out .get ('ok')}"
        )
        self._log(summary_text)
        if not out.get("ok"):
            if name == "roam_infer":
                print(
                    f" ROAM Error Info: {out .get ('error')or out .get ('formatted')}", flush=True
                )
                try:
                    r0 = (out.get("records") or [{}])[0]
                    log_tail = (r0.get("log_tail") or "").strip()
                    if log_tail:
                        print("----- ROAM Log Tail -----", flush=True)
                        print(log_tail[-3000:], flush=True)
                        print("-------------------------", flush=True)
                except Exception:
                    pass
            elif name == "raspr_run":
                print(
                    f" RaSPr Error Info: {out .get ('error')or out .get ('formatted')}", flush=True
                )
                try:
                    r0 = (out.get("records") or [{}])[0]
                    err_excerpt = (r0.get("error_excerpt") or "").strip()
                    log_tail = (r0.get("log_tail") or "").strip()
                    if err_excerpt:
                        print("----- RaSPr Error Excerpt -----", flush=True)
                        print(err_excerpt[-3000:], flush=True)
                        print("--------------------------------", flush=True)
                    if log_tail:
                        print("----- RaSPr Log Tail -----", flush=True)
                        print(log_tail[-3000:], flush=True)
                        print("--------------------------", flush=True)
                except Exception:
                    pass
        try:
            ev = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tool": name,
                "reason": reason,
                "params": args_clean,
                "duration_s": float(dt),
                "ok": bool(out.get("ok")),
                "records": int(len(out.get("records", []) or [])),
                "error": out.get("error") or "",
                "result_summary": summary_one_line,
            }
            if isinstance(out, dict) and out.get("records"):
                r0 = (out.get("records") or [{}])[0] or {}
                if isinstance(r0, dict):
                    for k in [
                        "return_code",
                        "outdir",
                        "output_dir",
                        "results_root",
                        "tiff_wsl",
                        "dicom_dir_wsl",
                    ]:
                        if k in r0:
                            ev[k] = r0.get(k)
            self._tool_events.append(ev)
        except Exception:
            pass
        return out

    def _dispatch_tool(self, name: str, args_clean: Dict[str, Any]) -> ToolResult:
        if name == "llm_mri":
            input_path = str(
                args_clean.get("input_path")
                or args_clean.get("path")
                or args_clean.get("image_path")
                or ""
            )
            imgs = args_clean.get("images")
            images_list: Optional[List[str]] = None
            if isinstance(imgs, list) and imgs:
                images_list = [str(x) for x in imgs if str(x).strip()]
                if (not input_path) and images_list:
                    input_path = images_list[0]
            return self._call_llm_mri_compat(
                input_path=input_path,
                images=images_list,
                question=str(args_clean.get("question") or args_clean.get("query") or ""),
                max_images=int(args_clean.get("max_images", 12) or 12),
                model=args_clean.get("model"),
            )
        elif name == "search_local_guidelines":
            return search_local_guidelines(str(args_clean.get("query", "")))
        elif name == "search_local_facts":
            return search_local_facts(str(args_clean.get("query", "")))
        elif name == "search_google":
            return search_google(str(args_clean.get("query", "")))
        elif name == "search_pubmed":
            return search_pubmed(str(args_clean.get("query", "")))
        elif name == "read_webpage":
            return read_webpage(str(args_clean.get("url", "")))
        elif name == "gigatime_infer":
            return gigatime_infer(
                input_tiff=str(args_clean.get("input_tiff", "")),
                output_dir=str(args_clean.get("output_dir", "")),
                conda_env=str(
                    args_clean.get("conda_env")
                    or os.environ.get("BIOPLEX_GIGATIME_CONDA_ENV")
                    or "gigatime_check_1"
                ),
                timeout_min=int(args_clean.get("timeout_min", 60) or 60),
                run_llm_analysis=bool(args_clean.get("run_llm_analysis", True)),
                llm_question=str(args_clean.get("llm_question", "")),
                llm_max_images=int(args_clean.get("llm_max_images", 12) or 12),
                llm_model=args_clean.get("llm_model"),
            )
        elif name == "llm_gigatime":
            return llm_gigatime(
                output_dir=str(args_clean.get("output_dir", "")),
                visualizations_dir=str(args_clean.get("visualizations_dir", "")),
                question=str(args_clean.get("question", "")),
                max_images=int(args_clean.get("max_images", 12) or 12),
                model=args_clean.get("model"),
            )
        elif name == "roam_infer":
            return roam_infer(
                input_tiff=str(args_clean.get("input_tiff", "")),
                wsl_distro=str(
                    args_clean.get("wsl_distro") or os.environ.get("BIOPLEX_WSL_DISTRO") or "Ubuntu"
                ),
                timeout_min=int(args_clean.get("timeout_min", 60) or 60),
                convert_windows_path_to_wsl=bool(
                    args_clean.get("convert_windows_path_to_wsl", True)
                ),
                roam_root=str(args_clean.get("roam_root", "")),
                assets_root=str(args_clean.get("assets_root", "")),
                results_root=str(args_clean.get("results_root", "")),
                conda_env=str(
                    args_clean.get("conda_env") or os.environ.get("BIOPLEX_ROAM_CONDA_ENV") or "roam"
                ),
            )
        elif name == "raspr_run":
            return raspr_run(
                case=str(args_clean.get("case", "")),
                dicom_dir=str(args_clean.get("dicom_dir", "")),
                output_dir=str(args_clean.get("output_dir", "")),
                timeout_min=int(args_clean.get("timeout_min", 120) or 120),
                script_path=str(args_clean.get("script_path", "")),
                wsl_distro=str(
                    args_clean.get("wsl_distro") or os.environ.get("BIOPLEX_WSL_DISTRO") or "Ubuntu"
                ),
                convert_windows_path_to_wsl=bool(
                    args_clean.get("convert_windows_path_to_wsl", True)
                ),
            )
        elif name == "scrna_pipeline_run":
            import re as _re

            def _fix_ext(p: str) -> str:
                if not p:
                    return p
                p = p.strip()
                p = _re.sub(r"\s*\.:h5ad$", ".h5ad", p, flags=_re.IGNORECASE)
                p = _re.sub(r"\s*\.:enx:$", ".h5ad", p, flags=_re.IGNORECASE)
                p = _re.sub(
                    r"\s+\.(pt|json|py|csv|txt)$",
                    lambda m: "." + m.group(1),
                    p,
                    flags=_re.IGNORECASE,
                )
                return p

            import os as _os

            _PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
            _tools_dir_default_raw = (os.environ.get("BIOPLEX_SCRNA_TOOLS_DIR") or "").strip()
            if _tools_dir_default_raw:
                _tools_dir_default_path = Path(_tools_dir_default_raw)
                if not _tools_dir_default_path.is_absolute():
                    _tools_dir_default_path = _PROJECT_ROOT_DEFAULT / _tools_dir_default_path
                _TOOLS_DIR_DEFAULT = str(_tools_dir_default_path.resolve())
            else:
                _TOOLS_DIR_DEFAULT = str((_PROJECT_ROOT_DEFAULT / "tools" / "scrna").resolve())
            _tools_dir_raw = str(args_clean.get("tools_dir") or "").strip()
            if _tools_dir_raw:
                _tools_dir_path = Path(_tools_dir_raw)
                if not _tools_dir_path.is_absolute():
                    _tools_dir_path = _PROJECT_ROOT_DEFAULT / _tools_dir_path
                _tools_dir_candidate = str(_tools_dir_path.resolve())
            else:
                _tools_dir_candidate = ""
            if _tools_dir_candidate and _os.path.isdir(_tools_dir_candidate):
                tools_dir = _tools_dir_candidate
            else:
                if _tools_dir_raw:
                    print(
                        f" [scRNA][PATH-FIX] tools_dir message message {_tools_dir_raw !r }",
                        flush=True,
                    )
                tools_dir = _TOOLS_DIR_DEFAULT
            _raw_input = (
                str(args_clean.get("input_data") or "").strip()
                or str(args_clean.get("h5ad_path") or "").strip()
                or str(args_clean.get("tenx_dir") or "").strip()
            )
            _raw_input = _fix_ext(_raw_input)
            if _raw_input and _os.path.exists(_raw_input):
                input_data_raw = _raw_input
            elif _raw_input.lower().endswith(".h5ad"):
                _dir_candidate = _re.sub(r"\.h5ad$", "", _raw_input, flags=_re.IGNORECASE)
                if _os.path.isdir(_dir_candidate):
                    print(
                        f" [scRNA][PATH-FIX] .h5ad message message {_dir_candidate !r }", flush=True
                    )
                    input_data_raw = _dir_candidate
                else:
                    input_data_raw = _raw_input
            else:
                input_data_raw = _raw_input
            args_clean["input_data"] = input_data_raw
            print(f" [scRNA][PATH-CLEAN] input_data = {input_data_raw !r }", flush=True)
            print(f" [scRNA][PATH-CLEAN] tools_dir = {tools_dir !r }", flush=True)
            for _pk in ("scmulan_ckpt", "classifier_model", "classifier_info", "outdir"):
                if args_clean.get(_pk):
                    args_clean[_pk] = _fix_ext(str(args_clean[_pk]))
                    print(f" [scRNA][PATH-CLEAN] {_pk :20s}= {args_clean [_pk ]!r }", flush=True)
            if input_data_raw and not input_data_raw.lower().endswith(".h5ad"):
                try:
                    from pathlib import Path
                    import shutil

                    tenx_path = Path(input_data_raw)
                    if tenx_path.is_dir():
                        _10x_files = {
                            "barcodes.tsv",
                            "barcodes.tsv.gz",
                            "features.tsv",
                            "features.tsv.gz",
                            "genes.tsv",
                            "genes.tsv.gz",
                            "matrix.mtx",
                            "matrix.mtx.gz",
                        }
                        root_files = {f.name.lower() for f in tenx_path.iterdir() if f.is_file()}
                        has_10x_files_in_root = bool(root_files & _10x_files)
                        has_subdirs = any(f.is_dir() for f in tenx_path.iterdir())
                        if has_10x_files_in_root and not has_subdirs:
                            sample_dir = tenx_path / "sample1"
                            sample_dir.mkdir(exist_ok=True)
                            for f in list(tenx_path.iterdir()):
                                if f.is_file():
                                    shutil.move(str(f), str(sample_dir / f.name))
                            print(f" [scRNA][AUTO-FIX] 10x message: {sample_dir }", flush=True)
                        elif not has_10x_files_in_root and not has_subdirs:
                            print(f" [scRNA][WARN] tenx message: {tenx_path }", flush=True)
                except Exception as _fix_err:
                    print(f" [scRNA][WARN] 10x message {_fix_err } message", flush=True)

            def _looks_like_dir(p: str) -> bool:
                if not p:
                    return True
                ps = p.strip().strip('"').strip("'")
                if ps.endswith("\\") or ps.endswith("/"):
                    return True
                if ("." not in ps.split("\\")[-1]) and ("." not in ps.split("/")[-1]):
                    return True
                return False

            scmulan_ckpt_in = str(args_clean.get("scmulan_ckpt", "") or "").strip()
            classifier_model_in = str(args_clean.get("classifier_model", "") or "").strip()
            classifier_info_in = str(args_clean.get("classifier_info", "") or "").strip()
            scmulan_ckpt = scmulan_ckpt_in
            if (
                (not scmulan_ckpt)
                or _looks_like_dir(scmulan_ckpt)
                or (not scmulan_ckpt.lower().endswith(".pt"))
            ):
                scmulan_ckpt = str(Path(tools_dir) / "ckpt_scMulan.pt")
            classifier_model = classifier_model_in
            if (
                (not classifier_model)
                or _looks_like_dir(classifier_model)
                or (not classifier_model.lower().endswith(".pt"))
            ):
                classifier_model = str(Path(tools_dir) / "classifier_model.pt")
            classifier_info = classifier_info_in
            if (
                (not classifier_info)
                or _looks_like_dir(classifier_info)
                or (not classifier_info.lower().endswith(".json"))
            ):
                classifier_info = str(Path(tools_dir) / "classifier_info.json")
            try:
                if not Path(scmulan_ckpt).exists():
                    fallback = str(Path(tools_dir) / "ckpt_scMulan.pt")
                    print(
                        f" [scRNA][WARN] scmulan_ckpt not found: {scmulan_ckpt } -> fallback {fallback }",
                        flush=True,
                    )
                    scmulan_ckpt = fallback
                if not Path(classifier_model).exists():
                    fallback = str(Path(tools_dir) / "classifier_model.pt")
                    print(
                        f" [scRNA][WARN] classifier_model not found: {classifier_model } -> fallback {fallback }",
                        flush=True,
                    )
                    classifier_model = fallback
                if not Path(classifier_info).exists():
                    fallback = str(Path(tools_dir) / "classifier_info.json")
                    print(
                        f" [scRNA][WARN] classifier_info not found: {classifier_info } -> fallback {fallback }",
                        flush=True,
                    )
                    classifier_info = fallback
            except Exception:
                pass
            print(f" [scRNA] tools_dir: {tools_dir }", flush=True)
            print(f" [scRNA] scmulan_ckpt: {scmulan_ckpt }", flush=True)
            print(f" [scRNA] classifier_model: {classifier_model }", flush=True)
            print(f" [scRNA] classifier_info: {classifier_info }", flush=True)
            return scrna_pipeline_run(
                input_data=str(args_clean.get("input_data", "")),
                outdir=str(args_clean.get("outdir", "")),
                scmulan_ckpt=scmulan_ckpt,
                classifier_model=classifier_model,
                classifier_info=classifier_info,
                sample_id=str(args_clean.get("sample_id", "MySample")),
                conda_env=str(
                    args_clean.get("conda_env") or os.environ.get("BIOPLEX_SCRNA_CONDA_ENV") or "base"
                ),
                tools_dir=tools_dir,
                pipeline_script=str(
                    args_clean.get("pipeline_script")
                    or os.environ.get("BIOPLEX_SCRNA_EXTERNAL_PIPELINE")
                    or "integrated_analysis_pipeline.py"
                ),
                timeout_min=int(args_clean.get("timeout_min", 240) or 240),
            )
        else:
            return {
                "ok": False,
                "tool_name": name,
                "records": [],
                "formatted": "",
                "error": f"Unknown tool: {name }",
            }

    _NON_CITATION_TOOLS = frozenset(
        {
            "llm_mri",
            "raspr_run",
            "gigatime_infer",
            "llm_gigatime",
            "roam_infer",
            "scrna_pipeline_run",
        }
    )

    def _collect_citations_from_tool(
        self, tool_name: str, tool_result: ToolResult
    ) -> List[Citation]:
        if tool_name in self._NON_CITATION_TOOLS:
            return []
        citations: List[Citation] = []
        if tool_result and tool_result.get("records"):
            for rec in tool_result["records"]:
                if not isinstance(rec, dict):
                    continue
                title = rec.get("title") or ""
                pmid = rec.get("pmid") or ""
                url = rec.get("url") or ""
                if not (title or pmid or url):
                    continue
                citations.append(
                    {
                        "title": title or "Status update.",
                        "year": rec.get("year"),
                        "url": url,
                        "pmid": pmid,
                        "source": rec.get("source") or tool_name,
                    }
                )
        return citations

    def run(
        self,
        query: str,
        plan: Plan,
        model_name: Optional[str] = None,
        max_steps: int = MAX_REACT_STEPS,
    ) -> Tuple[Dict[str, ToolResult], List[Citation], str]:
        self._log_step("Status update.")
        executed_tools: List[str] = []
        all_citations: List[Citation] = []
        tool_outputs: Dict[str, ToolResult] = {}
        draft_answer: str = ""
        plan_hint = ""
        required_tools_msg = ""
        if plan and plan.suggested_tools:
            plan_hint = "Status update." + " -> ".join(plan.suggested_tools) + "\n"
            required_tools_msg = " ".join(plan.suggested_tools)
        numbered_questions = _extract_numbered_questions(query)
        numbered_questions_text = ""
        if numbered_questions:
            numbered_questions_text = (
                "Status update." + "\n".join([f"- {x }" for x in numbered_questions]) + "\n"
            )
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_01", locals())
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query},
        ]
        for step in range(max_steps):
            sys.stdout.flush()
            sys.stderr.flush()
            self._log_step(f"[PipelineAgent] message [message {step +1 }/{max_steps }]")
            print(f"\n [PipelineAgent] message {step +1 }/{max_steps } message...", flush=True)
            sys.stdout.flush()
            print(f" [PipelineAgent] message...", flush=True)
            total_size = sum(len(str(m.get("content") or "")) for m in messages)
            print(f" message {total_size } message {len (messages )} message ", flush=True)
            if total_size > 80000:
                sys_user = messages[:2]
                tail = messages[-3:]
                middle = messages[2:-3]
                compressed_middle = []
                for m in middle:
                    if m.get("role") == "tool":
                        content = str(m.get("content") or "")
                        if len(content) > 200:
                            m = dict(m)
                            m["content"] = content[:200] + "Status update."
                    compressed_middle.append(m)
                messages = sys_user + compressed_middle + tail
                self._log(
                    f" [PipelineAgent] messages message {total_size }  {sum (len (str (m .get ('content')or ''))for m in messages )} message"
                )
            _msg_sizes = [len(str(m.get("content") or "")) for m in messages]
            print(
                f"  [PipelineAgent] message LLM message {len (messages )} message message {sum (_msg_sizes )} message ...",
                flush=True,
            )
            resp = self._chat_completion(
                messages,
                model_name=model_name,
                tools=self.tools_schema,
                tool_choice="auto",
                temperature=0.2,
            )
            print(f"  [PipelineAgent] LLM message message...", flush=True)
            msg = resp.choices[0].message
            messages.append(msg.model_dump())
            goal_text = (msg.content or "").strip()
            if goal_text:
                show = goal_text.replace("\n", " ")
                if len(show) > 160:
                    self._log(f"[PipelineAgent] message message {show [:160 ]}...")
                    draft_answer = goal_text
                    print(f" [PipelineAgent] message {show [:160 ]}...", flush=True)
                else:
                    self._log(f"[PipelineAgent] message {goal_text }")
                    print(f" [PipelineAgent] message {show }", flush=True)
            if not msg.tool_calls:
                if plan and plan.suggested_tools:
                    pending = [t for t in plan.suggested_tools if t not in executed_tools]
                    if pending and (step < max_steps - 1):
                        next_tool = pending[0]
                        warning_msg = (
                            f"SYSTEM COMMAND: message message {next_tool } \n"
                            f"message {next_tool } message "
                        )
                        self._log(f" [PipelineAgent] message message {next_tool } message ")
                        print(
                            f" [PipelineAgent] message message {next_tool } message message...",
                            flush=True,
                        )
                        messages.append({"role": "user", "content": warning_msg})
                        continue
                self._log("Status update.")
                print("Status update.", flush=True)
                break
            tool_calls = list(msg.tool_calls or [])
            _prio = {
                "scrna_pipeline_run": 0,
                "llm_mri": 1,
                "raspr_run": 2,
                "gigatime_infer": 3,
                "roam_infer": 4,
            }
            tool_calls.sort(key=lambda tc: _prio.get(tc.function.name, 99))
            self._log(f"[PipelineAgent] message {len (tool_calls )} message ")
            tool_names_this_round = [tc.function.name for tc in tool_calls]
            print(
                f"\n [PipelineAgent] message {len (tool_calls )} message {'  '.join (tool_names_this_round )}",
                flush=True,
            )
            for i, tc in enumerate(tool_calls, start=1):
                self._log(f" - {i }) {tc .function .name }")
            for tc_i, tc in enumerate(tool_calls, start=1):
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                reason = (args.get("reason") or "").strip()
                log_reason = reason if reason else "Status update."
                self._log(
                    f"message Step {self ._step_counter }.{tc_i } message {func_name } | message {log_reason }"
                )
                _EXPENSIVE_TOOLS = {
                    "raspr_run",
                    "gigatime_infer",
                    "roam_infer",
                    "scrna_pipeline_run",
                    "llm_mri",
                }
                if func_name in _EXPENSIVE_TOOLS and func_name in tool_outputs:
                    self._log(f" [PipelineAgent] message {func_name } message message message")
                    print(
                        f" [PipelineAgent] {func_name } message message message message...",
                        flush=True,
                    )
                    cached_result = tool_outputs[func_name]
                    cached_content = (cached_result.get("formatted") or "Status update.")[:500]
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": func_name,
                            "content": f"[message message] {cached_content }",
                        }
                    )
                    executed_tools.append(func_name)
                    continue
                tool_result = self._call_tool(func_name, args, plan=plan)
                executed_tools.append(func_name)
                _FORMATTED_LIMITS = {
                    "scrna_pipeline_run": 3000,
                    "raspr_run": 2000,
                    "gigatime_infer": 3000,
                    "roam_infer": 2000,
                    "llm_mri": 3000,
                    "llm_gigatime": 3000,
                }
                _fmt_limit = _FORMATTED_LIMITS.get(func_name, 5000)
                if tool_result.get("formatted") and len(tool_result["formatted"]) > _fmt_limit:
                    _orig_len = len(tool_result["formatted"])
                    tool_result = dict(tool_result)
                    tool_result["formatted"] = (
                        tool_result["formatted"][:_fmt_limit]
                        + f"\n...[message message {_orig_len } message]"
                    )
                    print(
                        f"  [{func_name }] formatted message {_orig_len }  {_fmt_limit } message",
                        flush=True,
                    )
                tool_outputs[func_name] = tool_result
                all_citations.extend(self._collect_citations_from_tool(func_name, tool_result))
                self._log(f"[PipelineAgent] message message {len (all_citations )}")
                _last_ok = tool_result.get("ok", False)
                _status_icon = "" if _last_ok else "Status update."
                print(
                    f" message {len (all_citations )} message | message {[t for t in executed_tools ]}",
                    flush=True,
                )
                print(
                    f" {_status_icon } {func_name } message {"Status update."if _last_ok else "Status update."}",
                    flush=True,
                )
                print(f"  message...", flush=True)
                sys.stdout.flush()
                raw_tool_content = tool_result.get("formatted") or "No content"
                tool_content = self._sanitize_str(raw_tool_content)[:2000]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": func_name,
                        "content": tool_content,
                    }
                )
        evidence_parts: List[str] = []
        for tool_name, result in tool_outputs.items():
            formatted = (result.get("formatted") or "").strip()
            if formatted:
                evidence_parts.append(f"=== {tool_name } message ===\n{formatted }")
        evidence_text = "\n\n".join(evidence_parts)
        self._log(
            f"[PipelineAgent] message message {len (tool_outputs )} message {len (all_citations )} message "
        )
        return tool_outputs, all_citations, evidence_text


class SpecialistAgent(LLMCallerMixin):
    AGENT_NAME: str = "SpecialistAgent"
    ROLE_DESC: str = "Status update."
    SYSTEM_PROMPT_TEMPLATE: str = "{role}"

    def __init__(
        self,
        model: str,
        trace_logs: List[str],
        step_counter_ref: List[int],
        stream_callback: Any = None,
    ) -> None:
        self.model = model
        self._trace_logs = trace_logs
        self._step_counter_ref = step_counter_ref
        self._stream_callback = stream_callback

    @property
    def _step_counter(self) -> int:
        return self._step_counter_ref[0]

    @_step_counter.setter
    def _step_counter(self, v: int) -> None:
        self._step_counter_ref[0] = v

    @staticmethod
    def _finalize_report(
        agent: "SpecialistAgent",
        messages: List[Dict[str, Any]],
        citations: List[Citation],
        supp_citations: List[Citation],
        model_name: Optional[str],
    ) -> AgentMessage:
        resp = agent._chat_completion(messages, model_name=model_name, temperature=0.3)
        content = (resp.choices[0].message.content or "").strip()
        content = agent._enforce_word_limit(content, limit=3000)
        agent._log(f"[{agent .AGENT_NAME }] message {len (content )} message message 3000message ")
        ref_section, all_cites = agent._build_formatted_references(
            pipeline_citations=citations,
            supplementary_citations=supp_citations,
            report_content=content,
            model_name=model_name,
        )
        final_content = content + "Status update." + ref_section
        return AgentMessage(
            agent_name=agent.AGENT_NAME,
            role="specialist_report",
            content=final_content,
            citations=all_cites,
        )

    def _build_system_prompt(self) -> str:
        base = self.SYSTEM_PROMPT_TEMPLATE.format(role=self.ROLE_DESC)
        if base == self.ROLE_DESC:
            base = specialist_system_prompt(self.ROLE_DESC)
        return base + SPECIALIST_INITIAL_ISOLATION_PROMPT

    def _build_discussion_system_prompt(self) -> str:
        base = self.SYSTEM_PROMPT_TEMPLATE.format(role=self.ROLE_DESC)
        if base == self.ROLE_DESC:
            base = specialist_system_prompt(self.ROLE_DESC)
        return base + SPECIALIST_DISCUSSION_PROMPT

    def generate_report(
        self,
        query: str,
        tool_outputs: Dict[str, ToolResult],
        evidence_text: str,
        citations: List[Citation],
        model_name: Optional[str] = None,
    ) -> AgentMessage:
        raise NotImplementedError(
            f"{self .__class__ .__name__ } message generate_report() " "Status update."
        )

    @staticmethod
    def _strip_references_section(text: str) -> str:
        markers = ["Status update.", "Status update.", "Status update."]
        for marker in markers:
            idx = text.find(marker)
            if idx != -1:
                return text[:idx].strip()
        return text

    @staticmethod
    def _truncate_bus_thread(bus_thread: str, max_chars: int = 1500) -> str:
        if not bus_thread or len(bus_thread) <= max_chars:
            return bus_thread or ""
        tail = bus_thread[-max_chars:]
        idx = tail.find(" ")
        if idx > 0:
            tail = tail[idx:]
        return "Status update." + tail

    def generate_discussion_turn(
        self,
        query: str,
        my_report: str,
        other_reports: str,
        bus_thread: str,
        round_num: int,
        model_name: Optional[str] = None,
    ) -> AgentMessage:
        self._log_step(f"[{self .AGENT_NAME }] message message {round_num } message ")
        my_report_body = self._strip_references_section(my_report or "")
        _my_report_safe = my_report_body[:600]
        _other_reports_safe = (other_reports or "")[:2000]
        _bus_thread_safe = self._truncate_bus_thread(bus_thread, max_chars=1500)
        self._log(
            f" [message] promptmessage my={len (_my_report_safe )} "
            f"other={len (_other_reports_safe )} bus={len (_bus_thread_safe )}"
        )
        prompt = render_ultra_prompt("ultra_prompt_prompt_01", locals())
        messages = [
            {"role": "system", "content": self._build_discussion_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        resp = self._chat_completion(messages, model_name=model_name, temperature=0.4)
        content = (resp.choices[0].message.content or "").strip()
        self._log(f"[{self .AGENT_NAME }] message {round_num } message {len (content )} message ")
        return AgentMessage(
            agent_name=self.AGENT_NAME,
            role="discussion",
            content=content,
            metadata={"round": round_num},
        )

    @staticmethod
    def _build_citation_index(citations: List[Citation]) -> str:
        if not citations:
            return "Status update."
        lines: List[str] = []
        seen = set()
        idx = 1
        for c in citations:
            key = (c.get("pmid"), c.get("title"), c.get("url"))
            if key in seen:
                continue
            seen.add(key)
            title = c.get("title") or "Status update."
            meta = []
            if c.get("source"):
                meta.append(f"Source={c ['source']}")
            if c.get("year"):
                meta.append(f"Year={c ['year']}")
            if c.get("pmid"):
                meta.append(f"PMID={c ['pmid']}")
            line = f"[{idx }] {title }"
            if meta:
                line += f" ({'; '.join (meta )})"
            lines.append(line)
            if c.get("url"):
                lines.append(f" {c ['url']}")
            idx += 1
        return "\n".join(lines)

    def _decide_search_queries(
        self,
        query: str,
        evidence_text: str,
        model_name: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        role = self.ROLE_DESC
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_02", locals())
        user_content = render_ultra_prompt("ultra_user_content_prompt_02", locals())
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        default: Dict[str, List[str]] = {"local_guidelines": [], "pubmed": [], "google": []}
        try:
            resp = self._chat_completion(messages, model_name=model_name, temperature=0.1)
            raw = (resp.choices[0].message.content or "").strip()
            cleaned = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(cleaned)
            result: Dict[str, List[str]] = {
                "local_guidelines": (data.get("local_guidelines") or [])[:3],
                "pubmed": (data.get("pubmed") or [])[:3],
                "google": (data.get("google") or [])[:2],
            }
            self._log(f"[{self .AGENT_NAME }] message {result }")
            return result
        except Exception as e:
            self._log(f"[{self .AGENT_NAME }] message {e } message")
            return default

    def _run_supplementary_search(
        self,
        search_queries: Dict[str, List[str]],
    ) -> Tuple[List[Citation], str]:
        new_citations: List[Citation] = []
        evidence_blocks: List[str] = []
        for q in search_queries.get("local_guidelines", []):
            if not q.strip():
                continue
            self._log(f" [message][local_guidelines] {q }")
            try:
                res = search_local_guidelines(q)
                if res.get("ok") and res.get("records"):
                    for rec in (res.get("records") or [])[:4]:
                        raw_content = (
                            rec.get("content")
                            or rec.get("text")
                            or rec.get("abstract")
                            or rec.get("chunk")
                            or ""
                        ).strip()
                        title = rec.get("title") or raw_content[:60] or "Status update."
                        c: Citation = {
                            "title": title,
                            "year": rec.get("year"),
                            "url": rec.get("url"),
                            "pmid": None,
                            "source": "local_guidelines",
                            "_search_query": q,
                            "_specialist": self.AGENT_NAME,
                            "_content": raw_content[:1200],
                        }
                        new_citations.append(c)
                    evidence_blocks.append(
                        f"[message: {q }]\n" + (res.get("formatted") or "")[:800]
                    )
            except Exception as e:
                self._log(f" [message][local_guidelines] message {e }")
        for q in search_queries.get("pubmed", []):
            if not q.strip():
                continue
            self._log(f" [message][pubmed] {q }")
            try:
                res = search_pubmed(q)
                if res.get("ok") and res.get("records"):
                    for rec in (res.get("records") or [])[:5]:
                        pmid = rec.get("pmid") or ""
                        raw_content = (
                            rec.get("abstract")
                            or rec.get("text")
                            or rec.get("content")
                            or rec.get("summary")
                            or ""
                        ).strip()
                        if not raw_content:
                            formatted = res.get("formatted") or ""
                            if pmid and pmid in formatted:
                                idx = formatted.find(pmid)
                                raw_content = formatted[idx : idx + 400].strip()
                        c = {
                            "title": rec.get("title") or "Status update.",
                            "year": rec.get("year"),
                            "url": (
                                rec.get("url")
                                or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid }/" if pmid else None)
                            ),
                            "pmid": pmid,
                            "source": "pubmed",
                            "_search_query": q,
                            "_specialist": self.AGENT_NAME,
                            "_content": raw_content[:1200],
                        }
                        new_citations.append(c)
                    evidence_blocks.append(
                        f"[PubMedmessage: {q }]\n" + (res.get("formatted") or "")[:800]
                    )
            except Exception as e:
                self._log(f" [message][pubmed] message {e }")
        for q in search_queries.get("google", []):
            if not q.strip():
                continue
            self._log(f" [message][google] {q }")
            try:
                res = search_google(q)
                if res.get("ok") and res.get("records"):
                    for rec in (res.get("records") or [])[:3]:
                        raw_content = (
                            rec.get("snippet")
                            or rec.get("abstract")
                            or rec.get("text")
                            or rec.get("content")
                            or ""
                        ).strip()
                        c = {
                            "title": rec.get("title") or "Status update.",
                            "year": None,
                            "url": rec.get("link") or rec.get("url"),
                            "pmid": None,
                            "source": "google",
                            "_search_query": q,
                            "_specialist": self.AGENT_NAME,
                            "_content": raw_content[:1200],
                        }
                        new_citations.append(c)
                    evidence_blocks.append(
                        f"[Googlemessage: {q }]\n" + (res.get("formatted") or "")[:800]
                    )
            except Exception as e:
                self._log(f" [message][google] message {e }")
        supplementary_text = "\n\n".join(evidence_blocks)
        self._log(f"[{self .AGENT_NAME }] message message {len (new_citations )} message")
        return new_citations, supplementary_text

    @staticmethod
    def _enforce_word_limit(content: str, limit: int = 1000) -> str:
        body = content
        suffix = ""
        for marker in ["Status update.", "Status update.", "Status update."]:
            idx = content.find(marker)
            if idx != -1:
                body = content[:idx]
                suffix = content[idx:]
                break
        if len(body) <= limit:
            return content
        truncated = body[:limit]
        for punct in (" ", " ", ".", ";", "\n"):
            last = truncated.rfind(punct)
            if last > limit * 0.7:
                truncated = truncated[: last + 1]
                break
        truncated = truncated.rstrip() + "Status update."
        return truncated + suffix

    def _build_formatted_references(
        self,
        pipeline_citations: List[Citation],
        supplementary_citations: List[Citation],
        report_content: str,
        model_name: Optional[str] = None,
    ) -> Tuple[str, List[Citation]]:
        _EXCLUDED_SOURCES = {
            "llm_mri",
            "raspr_run",
            "gigatime_infer",
            "llm_gigatime",
            "roam_infer",
            "scrna_pipeline_run",
        }
        all_cites: List[Citation] = []
        seen_keys: set = set()
        for c in pipeline_citations + supplementary_citations:
            source = (c.get("source") or "").lower()
            if source in _EXCLUDED_SOURCES:
                continue
            title = c.get("title") or ""
            if title.startswith("Tool output:"):
                continue
            key = (c.get("pmid"), title, c.get("url"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            all_cites.append(c)
        if not all_cites:
            return "Status update.", []
        role = self.ROLE_DESC
        cite_list_text = ""
        for i, c in enumerate(all_cites, start=1):
            title = (c.get("title") or "Status update.")[:100]
            source = c.get("source") or "N/A"
            pmid = c.get("pmid") or ""
            url = c.get("url") or ""
            specialist = c.get("_specialist") or ""
            query_hint = c.get("_search_query") or ""
            content_raw = (c.get("_content") or "").strip()
            cite_list_text += f"[{i }]\n"
            cite_list_text += f" message {title }\n"
            cite_list_text += f" message {source }"
            if pmid:
                cite_list_text += f" | PMID={pmid }"
            if url:
                cite_list_text += f" | URL={url }"
            if query_hint:
                cite_list_text += f" | message {query_hint }"
            if specialist:
                cite_list_text += f" | message {specialist }"
            cite_list_text += "\n"
            if content_raw:
                cite_list_text += f" message {content_raw [:600 ]}\n"
            else:
                cite_list_text += f" message message message \n"
            cite_list_text += "\n"
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_03", locals())
        user_content = render_ultra_prompt("ultra_user_content_prompt_03", locals())
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            resp = self._chat_completion(messages, model_name=model_name, temperature=0.2)
            ref_section = (resp.choices[0].message.content or "").strip()
            self._log(f"[{self .AGENT_NAME }] message {len (all_cites )} message ")
            return ref_section, all_cites
        except Exception as e:
            self._log(f"[{self .AGENT_NAME }] message {e } message")
            fallback_lines = []
            for i, c in enumerate(all_cites, start=1):
                title = (c.get("title") or "Status update.")[:100]
                source = c.get("source") or "N/A"
                pmid = c.get("pmid") or ""
                url = c.get("url") or ""
                content_raw = (c.get("_content") or "").strip()
                header = (
                    f"message {i } {title }"
                    + (f" {source }, PMID={pmid } " if pmid else f" {source } ")
                    + (f"\n URL {url }" if url else "")
                )
                content_line = f"message {content_raw [:400 ]}" if content_raw else "Status update."
                fallback_lines.append(f"{header }\n{content_line }\nmessage message message ")
            return "\n\n".join(fallback_lines), all_cites


class ImagingAgent(SpecialistAgent):
    AGENT_NAME = "ImagingAgent"
    ROLE_DESC = "Status update."
    OWNED_TOOLS: List[str] = ["llm_mri", "raspr_run"]
    SYSTEM_PROMPT_TEMPLATE = SPECIALIST_ROLE_PROMPTS["imaging"]

    def generate_report(
        self,
        query: str,
        tool_outputs: Dict[str, ToolResult],
        evidence_text: str,
        citations: List[Citation],
        model_name: Optional[str] = None,
    ) -> AgentMessage:
        mri_output = (tool_outputs.get("llm_mri") or {}).get("formatted", "") or ""
        raspr_output = (tool_outputs.get("raspr_run") or {}).get("formatted", "") or ""
        self._log_step(f"[{self .AGENT_NAME }] message")
        sq = self._decide_search_queries(
            query=query, evidence_text=evidence_text, model_name=model_name
        )
        supp_cites, supp_ev = self._run_supplementary_search(sq)
        tool_block = (
            f" llm_mri message message  \n"
            f"{mri_output if mri_output else "Status update."}\n\n"
            f" RaSPr message message  \n"
            f"{raspr_output if raspr_output else "Status update."}"
        )
        lit_block = evidence_text[:1500] + ("Status update." + supp_ev if supp_ev else "")
        ref_text = self._build_citation_index(citations + supp_cites)
        user_content = render_ultra_prompt("ultra_user_content_prompt_04", locals())
        msgs = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        return SpecialistAgent._finalize_report(self, msgs, citations, supp_cites, model_name)


class PathologyAgent(SpecialistAgent):
    AGENT_NAME = "PathologyAgent"
    ROLE_DESC = "Status update."
    OWNED_TOOLS: List[str] = ["roam_infer", "gigatime_infer", "llm_gigatime"]
    SYSTEM_PROMPT_TEMPLATE = SPECIALIST_ROLE_PROMPTS["pathology"]

    def generate_report(
        self,
        query: str,
        tool_outputs: Dict[str, ToolResult],
        evidence_text: str,
        citations: List[Citation],
        model_name: Optional[str] = None,
    ) -> AgentMessage:
        roam_output = (tool_outputs.get("roam_infer") or {}).get("formatted", "") or ""
        gt_res = tool_outputs.get("gigatime_infer") or {}
        gigatime_output = gt_res.get("formatted", "") or ""
        llm_gigatime_output = ""
        gt_recs = gt_res.get("records") or []
        if gt_recs:
            llm_block = (gt_recs[0] or {}).get("llm_gigatime") or {}
            if llm_block.get("ok"):
                llm_gigatime_output = (llm_block.get("formatted") or "").strip()
        if not llm_gigatime_output:
            llm_gigatime_output = (tool_outputs.get("llm_gigatime") or {}).get(
                "formatted", ""
            ) or ""
        self._log_step(f"[{self .AGENT_NAME }] message")
        sq = self._decide_search_queries(
            query=query, evidence_text=evidence_text, model_name=model_name
        )
        supp_cites, supp_ev = self._run_supplementary_search(sq)
        tool_block = (
            f" ROAM message message  \n"
            f"{roam_output if roam_output else "Status update."}\n\n"
            f" GigaTIME message message  \n"
            f"{gigatime_output if gigatime_output else "Status update."}\n\n"
            f" llm_gigatime Map message message  \n"
            f"{llm_gigatime_output if llm_gigatime_output else "Status update."}"
        )
        lit_block = evidence_text[:1500] + ("Status update." + supp_ev if supp_ev else "")
        ref_text = self._build_citation_index(citations + supp_cites)
        user_content = render_ultra_prompt("ultra_user_content_prompt_05", locals())
        msgs = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        return SpecialistAgent._finalize_report(self, msgs, citations, supp_cites, model_name)


class OncologyAgent(SpecialistAgent):
    AGENT_NAME = "OncologyAgent"
    ROLE_DESC = "Status update."
    OWNED_TOOLS: List[str] = ["scrna_pipeline_run"]
    _SCRNA_PRIOR_KNOWLEDGE_TEXT = (
        "Status update."
        "Status update."
        "Status update."
        "Status update."
        "Status update."
        "Status update."
        "FOSTAMATINIB AMINOPURVALANOL-A TERAZOSIN \n"
        "Status update."
        "Status update."
    )
    _SCRNA_CLUSTER_INTERP = {
        "cluster0": "Status update.",
        "cluster1": "Status update.",
        "cluster2": "Status update.",
        "cluster3": "Status update.",
        "0": "Status update.",
        "1": "Status update.",
        "2": "Status update.",
        "3": "Status update.",
    }
    _SCRNA_DRUG_HINT_CLUSTER3 = "Status update."
    SYSTEM_PROMPT_TEMPLATE = SPECIALIST_ROLE_PROMPTS["oncology"]

    def generate_report(
        self,
        query: str,
        tool_outputs: Dict[str, ToolResult],
        evidence_text: str,
        citations: List[Citation],
        model_name: Optional[str] = None,
    ) -> AgentMessage:
        scrna_res = tool_outputs.get("scrna_pipeline_run") or {}
        scrna_output = scrna_res.get("formatted", "") or ""
        recs = scrna_res.get("records") or []
        r0 = recs[0] if recs else {}
        prediction_guess = str(r0.get("prediction_guess") or "")
        predicted_cluster = str(r0.get("predicted_cluster") or "")
        effective_cluster = predicted_cluster or prediction_guess
        cl = _normalize_scrna_cluster_label(effective_cluster)
        cluster_interp = _SCRNA_CLUSTER_INTERP.get(cl, "") if cl else ""
        drug_hint = _SCRNA_DRUG_HINT_CLUSTER3 if cl == "cluster3" else ""
        self._log_step(f"[{self .AGENT_NAME }] message")
        sq = self._decide_search_queries(
            query=query, evidence_text=evidence_text, model_name=model_name
        )
        supp_cites, supp_ev = self._run_supplementary_search(sq)
        tool_block = (
            f" scrna_pipeline_run message message  \n"
            f"{scrna_output if scrna_output else "Status update."}\n\n"
            f" message \n{_SCRNA_PRIOR_KNOWLEDGE_TEXT }\n"
            f" message CSV  {predicted_cluster or "Status update."}\n"
            f" message {prediction_guess or "Status update."}\n"
            f" message {effective_cluster or "Status update."}\n"
            f" message {cluster_interp or "Status update."}\n"
            f" message messagecluster3  {drug_hint or "Status update."}"
        )
        lit_block = evidence_text[:1500] + ("Status update." + supp_ev if supp_ev else "")
        ref_text = self._build_citation_index(citations + supp_cites)
        user_content = render_ultra_prompt("ultra_user_content_prompt_06", locals())
        msgs = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        result = SpecialistAgent._finalize_report(self, msgs, citations, supp_cites, model_name)
        result.metadata.update(
            {
                "prediction_guess": prediction_guess,
                "predicted_cluster": predicted_cluster,
                "effective_cluster": effective_cluster,
                "cluster": cl,
            }
        )
        return result


_SCRNA_PRIOR_KNOWLEDGE_TEXT = OncologyAgent._SCRNA_PRIOR_KNOWLEDGE_TEXT
_SCRNA_CLUSTER_INTERP = OncologyAgent._SCRNA_CLUSTER_INTERP
_SCRNA_DRUG_HINT_CLUSTER3 = OncologyAgent._SCRNA_DRUG_HINT_CLUSTER3


class NeurosurgeryAgent(SpecialistAgent):
    AGENT_NAME = "NeurosurgeryAgent"
    ROLE_DESC = "Status update."
    OWNED_TOOLS: List[str] = []
    SYSTEM_PROMPT_TEMPLATE = SPECIALIST_ROLE_PROMPTS["neurosurgery"]

    def generate_report(
        self,
        query: str,
        tool_outputs: Dict[str, ToolResult],
        evidence_text: str,
        citations: List[Citation],
        model_name: Optional[str] = None,
    ) -> AgentMessage:
        self._log_step(f"[{self .AGENT_NAME }] message")
        sq = self._decide_search_queries(
            query=query, evidence_text=evidence_text, model_name=model_name
        )
        supp_cites, supp_ev = self._run_supplementary_search(sq)
        lit_block = evidence_text[:2000] + ("Status update." + supp_ev if supp_ev else "")
        ref_text = self._build_citation_index(citations + supp_cites)
        user_content = render_ultra_prompt("ultra_user_content_prompt_07", locals())
        msgs = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        return SpecialistAgent._finalize_report(self, msgs, citations, supp_cites, model_name)


class RadiotherapyAgent(SpecialistAgent):
    AGENT_NAME = "RadiotherapyAgent"
    ROLE_DESC = "Status update."
    OWNED_TOOLS: List[str] = []
    SYSTEM_PROMPT_TEMPLATE = SPECIALIST_ROLE_PROMPTS["radiotherapy"]

    def generate_report(
        self,
        query: str,
        tool_outputs: Dict[str, ToolResult],
        evidence_text: str,
        citations: List[Citation],
        model_name: Optional[str] = None,
    ) -> AgentMessage:
        self._log_step(f"[{self .AGENT_NAME }] message")
        sq = self._decide_search_queries(
            query=query, evidence_text=evidence_text, model_name=model_name
        )
        supp_cites, supp_ev = self._run_supplementary_search(sq)
        lit_block = evidence_text[:2000] + ("Status update." + supp_ev if supp_ev else "")
        ref_text = self._build_citation_index(citations + supp_cites)
        user_content = render_ultra_prompt("ultra_user_content_prompt_08", locals())
        msgs = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        return SpecialistAgent._finalize_report(self, msgs, citations, supp_cites, model_name)


class NursingAgent(SpecialistAgent):
    AGENT_NAME = "NursingAgent"
    ROLE_DESC = "Status update."
    OWNED_TOOLS: List[str] = []
    SYSTEM_PROMPT_TEMPLATE = SPECIALIST_ROLE_PROMPTS["nursing"]

    def generate_report(
        self,
        query: str,
        tool_outputs: Dict[str, ToolResult],
        evidence_text: str,
        citations: List[Citation],
        model_name: Optional[str] = None,
    ) -> AgentMessage:
        self._log_step(f"[{self .AGENT_NAME }] message")
        sq = self._decide_search_queries(
            query=query, evidence_text=evidence_text, model_name=model_name
        )
        supp_cites, supp_ev = self._run_supplementary_search(sq)
        lit_block = evidence_text[:2000] + ("Status update." + supp_ev if supp_ev else "")
        ref_text = self._build_citation_index(citations + supp_cites)
        user_content = render_ultra_prompt("ultra_user_content_prompt_09", locals())
        msgs = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        return SpecialistAgent._finalize_report(self, msgs, citations, supp_cites, model_name)


class RiskAuditorAgent(LLMCallerMixin):
    AGENT_NAME = "RiskAuditorAgent"

    def __init__(
        self,
        model: str,
        trace_logs: List[str],
        step_counter_ref: List[int],
        stream_callback: Any = None,
    ) -> None:
        self.model = model
        self._trace_logs = trace_logs
        self._step_counter_ref = step_counter_ref
        self._stream_callback = stream_callback

    @property
    def _step_counter(self) -> int:
        return self._step_counter_ref[0]

    @_step_counter.setter
    def _step_counter(self, v: int) -> None:
        self._step_counter_ref[0] = v

    def review_discussion_turn(
        self,
        query: str,
        round_num: int,
        round_messages: List[AgentMessage],
        model_name: Optional[str] = None,
    ) -> AgentMessage:
        self._log_step(f"[RiskAuditorAgent] message {round_num } message")
        round_text = ""
        for m in round_messages:
            brief = m.content[:300] + "..." if len(m.content) > 300 else m.content
            round_text += f" {m .agent_name } {brief }\n\n"
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_04", locals())
        user_content = render_ultra_prompt("ultra_user_content_prompt_10", locals())
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            resp = self._chat_completion(messages, model_name=model_name, temperature=0.1)
            review_content = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            review_content = f" RiskAuditorAgent message {round_num } message {e } "
        self._log(
            f"[RiskAuditorAgent] message {round_num } message {len (review_content )} message "
        )
        return AgentMessage(
            agent_name="RiskAuditorAgent",
            role="discussion_risk_review",
            content=review_content,
            metadata={"round": round_num, "type": "risk_review"},
        )

    def audit(
        self,
        draft_synthesis: str,
        discussion_thread: str,
        query: str,
        attempt: int = 1,
        model_name: Optional[str] = None,
    ) -> Tuple[bool, str, str]:
        self._log_step(f"[RiskAuditorAgent] message message {attempt } message ")
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_05", locals())
        user_content = render_ultra_prompt("ultra_user_content_prompt_11", locals())
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        resp = self._chat_completion(messages, model_name=model_name, temperature=0.1)
        audit_text = (resp.choices[0].message.content or "").strip()
        passed = False
        reason = "Status update."
        if "AUDIT_RESULT: PASS" in audit_text:
            passed = True
            reason = "Status update."
        elif "AUDIT_RESULT: FAIL" in audit_text:
            passed = False
            m = re.search(r"REASON:\s*(.+)", audit_text)
            reason = m.group(1).strip() if m else "Status update."
        self._log(f"[RiskAuditorAgent] message {' PASS'if passed else ' FAIL'} | {reason }")
        return passed, reason, audit_text


class SynthesizerAgent(LLMCallerMixin):
    AGENT_NAME = "SynthesizerAgent"

    def __init__(
        self,
        model: str,
        trace_logs: List[str],
        step_counter_ref: List[int],
        stream_callback: Any = None,
    ) -> None:
        self.model = model
        self._trace_logs = trace_logs
        self._step_counter_ref = step_counter_ref
        self._stream_callback = stream_callback

    @property
    def _step_counter(self) -> int:
        return self._step_counter_ref[0]

    @_step_counter.setter
    def _step_counter(self, v: int) -> None:
        self._step_counter_ref[0] = v

    def synthesize(
        self,
        query: str,
        specialist_reports: Dict[str, AgentMessage],
        discussion_bus: SharedBus,
        citations: List[Citation],
        numbered_questions: List[str],
        audit_feedback: Optional[str] = None,
        tool_outputs: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        attempt: int = 1,
    ) -> str:
        self._log_step(f"[SynthesizerAgent] message message {attempt } message ")
        _EXCLUDED_TOOL_SOURCES = {
            "llm_mri",
            "raspr_run",
            "gigatime_infer",
            "llm_gigatime",
            "roam_infer",
            "scrna_pipeline_run",
        }
        literature_citations = [
            c
            for c in citations
            if (c.get("source") or "").lower() not in _EXCLUDED_TOOL_SOURCES
            and not (c.get("title") or "").startswith("Tool output:")
            and not (c.get("url") or "").startswith("[llm_")
            and not (c.get("url") or "").startswith("[raspr")
            and not (c.get("url") or "").startswith("[scrna")
            and not (c.get("url") or "").startswith("[gigatime")
            and not (c.get("url") or "").startswith("[roam")
            and not (c.get("url") or "").startswith("[tool_")
        ]
        ref_text = SpecialistAgent._build_citation_index(literature_citations)
        self._log(
            f"[SynthesizerAgent] message {len (literature_citations )} message {len (citations )} message message "
        )
        reports_text = "\n\n".join(
            [
                f"=== {name } message ===\n{msg .content }"
                for name, msg in specialist_reports.items()
            ]
        )
        discussion_text = discussion_bus.get_full_thread_text()
        qlist_block = ""
        if numbered_questions:
            qlist_block = (
                "Status update."
                + "\n".join([f"{i +1 }. {x }" for i, x in enumerate(numbered_questions)])
                + "\n"
            )
        audit_block = ""
        if audit_feedback:
            _audit_reason = audit_feedback[:800].strip()
            audit_block = f"\n message message message  \n" f"{_audit_reason }\n" "Status update."
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_06", locals())
        _TOOL_KEYWORDS = [
            "Tool output:",
            "llm_mri",
            "raspr_run",
            "gigatime_infer",
            "llm_gigatime",
            "roam_infer",
            "scrna_pipeline_run",
            "[llm_",
            "[raspr",
            "[scrna",
            "[gigatime",
            "[roam",
            "[tool_",
            "local_imaging_report",
            "local_tool_output",
            "tool_output_dir",
        ]
        specialist_refs_text = ""
        for sp_name, sp_msg in specialist_reports.items():
            content_full = sp_msg.content or ""
            ref_block = ""
            for marker in ["Status update.", "Status update."]:
                if marker in content_full:
                    ref_block = content_full[content_full.find(marker) :]
                    break
            if ref_block:
                filtered_lines = []
                skip_entry = False
                for line in ref_block.split("\n"):
                    if line.strip().startswith("Status update."):
                        skip_entry = any(kw in line for kw in _TOOL_KEYWORDS)
                    if not skip_entry:
                        filtered_lines.append(line)
                filtered_block = "\n".join(filtered_lines)
                if filtered_block.strip():
                    specialist_refs_text += (
                        f"\n=== {sp_name } message ===\n{filtered_block [:2000 ]}\n"
                    )
        _query_short = query[:1000]
        _reports_short = reports_text[:4000]
        _refs_short = specialist_refs_text[:1000]
        _discussion_short = (discussion_text or "Status update.")[:1200]
        _ref_lines = ref_text.split("\n")
        _ref_limited = "\n".join(_ref_lines[:60])
        _total_est = (
            len(_query_short)
            + len(_reports_short)
            + len(_refs_short)
            + len(_discussion_short)
            + len(_ref_limited)
        )
        self._log(f"[SynthesizerAgent] user_content message {_total_est } message")
        print(f" [SynthesizerAgent] promptmessage {_total_est } message", flush=True)
        user_content = render_ultra_prompt("ultra_user_content_prompt_12", locals())
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        _FALLBACK_MODEL = GPT_MODEL
        _primary_model = model_name if (model_name and model_name != _FALLBACK_MODEL) else None
        _models_to_try: List[Tuple[str, float]] = []
        if _primary_model:
            _models_to_try.append((_primary_model, 600.0))
        _models_to_try.append((_FALLBACK_MODEL, 600.0))
        resp = None
        for _try_model, _timeout in _models_to_try:
            try:
                print(
                    f" [SynthesizerAgent] message {_try_model } timeout={_timeout }s ", flush=True
                )
                resp = self._chat_completion(
                    messages,
                    model_name=_try_model,
                    temperature=0.2,
                    max_tokens=32000,
                    timeout_override=_timeout,
                )
                _resp_text = (resp.choices[0].message.content or "").strip()
                _is_refused = (
                    len(_resp_text) < 50
                    or _resp_text.startswith("Status update.")
                    or _resp_text.startswith("Sorry")
                    or _resp_text.startswith("I cannot")
                )
                if _is_refused:
                    self._log(
                        f"[SynthesizerAgent] {_try_model } message/message {len (_resp_text )}message message"
                    )
                    print(f" [SynthesizerAgent] {_try_model } message message...", flush=True)
                    resp = None
                    continue
                print(
                    f" [SynthesizerAgent] message {_try_model } message {len (_resp_text )}message ",
                    flush=True,
                )
                break
            except Exception as _e:
                self._log(f"[SynthesizerAgent] message {_try_model } message {_e } message")
                print(
                    f" [SynthesizerAgent] {_try_model } message {type (_e ).__name__ } message...",
                    flush=True,
                )
                resp = None
                continue
        if resp is None:
            raise RuntimeError("Status update.")
        content = (resp.choices[0].message.content or "").strip()
        print(f" [SynthesizerAgent] LLMmessage {len (content )} message", flush=True)
        if content:
            preview = content[:120].replace("\n", " ")
            print(f" [SynthesizerAgent] message {preview }...", flush=True)
        else:
            print("Status update.", flush=True)
        _tool_outputs_safe = tool_outputs or {}
        _roam_section = ""
        if "roam_infer" in _tool_outputs_safe:
            _roam_res = _tool_outputs_safe["roam_infer"]
            _roam_recs = _roam_res.get("records") or []
            _roam_r0 = _roam_recs[0] if _roam_recs else {}
            _roam_pred = _roam_r0.get("predicted_subtype_guess") or "Status update."
            _roam_rc = _roam_r0.get("return_code", "N/A")
            _roam_ok = _roam_res.get("ok", False)
            _roam_section = (
                "\n\n---\n"
                "Status update."
                f"**message** {"Status update."if _roam_ok else "Status update."} return_code={_roam_rc })\n\n"
                f"**message** {_roam_pred }\n\n"
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
                "Status update."
            )
        else:
            _roam_section = (
                "\n\n---\n" "Status update." "Status update." "Status update." "Status update."
            )
        _gigatime_section = ""
        if "gigatime_infer" in _tool_outputs_safe:
            _gt_res = _tool_outputs_safe["gigatime_infer"]
            _gt_recs = _gt_res.get("records") or []
            _gt_r0 = _gt_recs[0] if _gt_recs else {}
            _gt_ok = _gt_res.get("ok", False)
            _gt_map_count = len(_gt_r0.get("map_images") or [])
            _gt_llm_block = _gt_r0.get("llm_gigatime") or {}
            _gt_llm_recs = _gt_llm_block.get("records") or []
            _gt_llm_r0 = _gt_llm_recs[0] if _gt_llm_recs else {}
            _gt_analysis = _gt_llm_r0.get("analysis") or {}
            _gt_overview = _gt_analysis.get("overview") or "Status update."
            _gt_notable = _gt_analysis.get("notable_markers") or []
            _gt_immune = _gt_analysis.get("immune_context") or "Status update."
            _gt_tumor = _gt_analysis.get("tumor_context") or "Status update."
            _notable_str = ""
            if _gt_notable:
                for _mk in _gt_notable[:6]:
                    _notable_str += f" - **{_mk .get ('marker','')}** message={_mk .get ('pattern','')} message={_mk .get ('intensity','')} {_mk .get ('notes','')}\n"
            else:
                _notable_str = "Status update."
            _gigatime_section = (
                "\n\n---\n"
                "Status update."
                f"**message** {"Status update."if _gt_ok else "Status update."}\n\n"
                f"**message** {_gt_map_count } message Map_*.png\n\n"
                f"**LLMmessage** {_gt_overview }\n\n"
                "Status update." + _notable_str + f"\n**message** {_gt_immune }\n\n"
                f"**message** {_gt_tumor }\n\n"
                "Status update."
                "Status update."
            )
        else:
            _gigatime_section = (
                "\n\n---\n" "Status update." "Status update." "Status update." "Status update."
            )
        content = content + _roam_section + _gigatime_section
        _DISCLAIMER = (
            "\n\n---\n"
            "Status update."
            "Status update."
            "Status update."
            "Status update."
            "Status update."
            "Status update."
        )
        if "Status update." not in content:
            content = content + _DISCLAIMER
        else:
            content = content + "Status update."
        content = SynthesizerAgent._renumber_references_globally(content)
        self._log(f"[SynthesizerAgent] message {len (content )} message message message ")
        return content

    @staticmethod
    def _renumber_references_globally(text: str) -> str:
        import re

        ref_entry_pat = re.compile(
            r"message",
            re.MULTILINE,
        )
        matches = list(ref_entry_pat.finditer(text))
        if not matches:
            return text
        seen_keys: dict = {}
        ordered_entries: list = []
        global_counter = 1
        for m in matches:
            full_entry = m.group(1)
            old_num = m.group(2)
            dedup_key = full_entry[:80].strip()
            if dedup_key not in seen_keys:
                seen_keys[dedup_key] = global_counter
                ordered_entries.append((old_num, global_counter, full_entry))
                global_counter += 1
            else:
                ordered_entries.append((old_num, seen_keys[dedup_key], full_entry))
        new_text = text
        replacements = []
        for m, (old_num, gnum, _) in zip(matches, ordered_entries):
            start, end = m.start(), m.end()
            new_entry = re.sub(
                r"message",
                lambda mo, n=gnum: f"{mo .group (1 )}{n }{mo .group (2 )}",
                m.group(1),
                count=1,
            )
            replacements.append((start, end, new_entry))
        for start, end, new_entry in reversed(replacements):
            new_text = new_text[:start] + new_entry + new_text[end:]
        old_to_new: dict = {}
        for old_num, gnum, _ in ordered_entries:
            if old_num not in old_to_new:
                old_to_new[old_num] = gnum
        sorted_old_nums = sorted(old_to_new.keys(), key=lambda x: int(x), reverse=True)
        for old_num in sorted_old_nums:
            gnum = old_to_new[old_num]
            if str(old_num) != str(gnum):
                new_text = re.sub(
                    r"\[" + re.escape(str(old_num)) + r"\]",
                    f"[{gnum }]",
                    new_text,
                )
        return new_text


class PlannerAgent(LLMCallerMixin):
    AGENT_NAME = "PlannerAgent"

    def __init__(
        self,
        model: str,
        tools_schema: List[Dict[str, Any]],
        trace_logs: List[str],
        step_counter_ref: List[int],
        stream_callback: Any = None,
    ) -> None:
        self.model = model
        self.tools_schema = tools_schema
        self._trace_logs = trace_logs
        self._step_counter_ref = step_counter_ref
        self._stream_callback = stream_callback

    @property
    def _step_counter(self) -> int:
        return self._step_counter_ref[0]

    @_step_counter.setter
    def _step_counter(self, v: int) -> None:
        self._step_counter_ref[0] = v

    def plan(
        self,
        query: str,
        static_plan: Plan,
        model_name: Optional[str] = None,
    ) -> Plan:
        self._log_step(f"[PlannerAgent] message")
        available_tools = [
            t["function"]["name"] for t in self.tools_schema if t.get("type") == "function"
        ]
        static_order = (
            " -> ".join(static_plan.suggested_tools)
            if static_plan.suggested_tools
            else "Status update."
        )
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_07", locals())
        user_content = render_ultra_prompt("ultra_user_content_prompt_13", locals())
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        print(f" [PlannerAgent] message LLM message model={model_name } ...", flush=True)
        try:
            resp = self._chat_completion(
                messages,
                model_name=model_name,
                temperature=0.1,
                timeout_override=120.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            plan_data: Dict[str, Any] = {}
            try:
                cleaned = re.sub(r"```json|```", "", raw).strip()
                plan_data = json.loads(cleaned)
            except Exception:
                self._log(f"[PlannerAgent] JSON message message")
                return static_plan
            tool_order = plan_data.get("tool_order") or static_plan.suggested_tools
            tool_order = [t for t in tool_order if t in available_tools]
            if not tool_order:
                tool_order = static_plan.suggested_tools
            branches = plan_data.get("conditional_branches") or []
            risk_flags = plan_data.get("risk_flags") or []
            planner_notes = plan_data.get("planner_notes") or ""
            notes_parts = [f"[PlannerAgent] {planner_notes }"]
            if branches:
                notes_parts.append("Status update.")
                for b in branches:
                    notes_parts.append(
                        f" - message {b .get ('condition')} message {b .get ('extra_tools')} | {b .get ('reason')}"
                    )
            if risk_flags:
                notes_parts.append("Status update." + " | ".join(risk_flags))
            enhanced_notes = "\n".join(notes_parts)
            self._log(f"[PlannerAgent] message {' -> '.join (tool_order )}")
            return Plan(
                parsed=static_plan.parsed,
                suggested_tools=tool_order,
                notes=enhanced_notes,
            )
        except Exception as e:
            self._log(f"[PlannerAgent] message {e } message")
            return static_plan


class LongitudinalMemoryAgent(LLMCallerMixin):
    AGENT_NAME = "LongitudinalMemoryAgent"

    def __init__(
        self,
        model: str,
        trace_logs: List[str],
        step_counter_ref: List[int],
        stream_callback: Any = None,
        memory_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self._trace_logs = trace_logs
        self._step_counter_ref = step_counter_ref
        self._stream_callback = stream_callback
        self.memory_dir = memory_dir or MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _step_counter(self) -> int:
        return self._step_counter_ref[0]

    @_step_counter.setter
    def _step_counter(self, v: int) -> None:
        self._step_counter_ref[0] = v

    def _get_patient_id(self, query: str) -> str:
        m = re.search(r"(MR\d{6,})", query, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m2 = re.search(r"message", query)
        if m2:
            return re.sub(r"[^\w]", "_", m2.group(1))
        import hashlib

        return "P_" + hashlib.md5(query[:20].encode()).hexdigest()[:8]

    def _get_memory_path(self, patient_id: str) -> Path:
        return self.memory_dir / f"{patient_id }_memory.json"

    def load_history(self, query: str) -> Tuple[str, str]:
        self._log_step(f"[LongitudinalMemoryAgent] message")
        patient_id = self._get_patient_id(query)
        mem_path = self._get_memory_path(patient_id)
        if not mem_path.exists():
            self._log(f"[LongitudinalMemoryAgent] message {patient_id } message message ")
            return patient_id, ""
        try:
            records = json.loads(mem_path.read_text(encoding="utf-8"))
            if not records:
                return patient_id, ""
            last = records[-1]
            history_lines = [
                f" message - message {patient_id } message {len (records )} message",
                f"message {last .get ('timestamp','N/A')}",
                f"message {last .get ('cluster','N/A')}",
                f"message {last .get ('treatment_summary','N/A')}",
                f"message {last .get ('recurrence_risk','N/A')}",
                f"message {last .get ('key_warnings','N/A')}",
            ]
            if len(records) >= 2:
                prev = records[-2]
                if (
                    prev.get("cluster")
                    and last.get("cluster")
                    and prev["cluster"] != last["cluster"]
                ):
                    history_lines.append(
                        f" message message={prev ['cluster']}  message={last ['cluster']} message "
                    )
            self._log(
                f"[LongitudinalMemoryAgent] message {patient_id } message {len (records )} message"
            )
            return patient_id, "\n".join(history_lines)
        except Exception as e:
            self._log(f"[LongitudinalMemoryAgent] message {e }")
            return patient_id, ""

    def save_record(
        self,
        patient_id: str,
        query: str,
        final_synthesis: str,
        specialist_reports: Dict[str, "AgentMessage"],
        tool_outputs: Dict[str, Any],
        model_name: Optional[str] = None,
    ) -> None:
        self._log_step(f"[LongitudinalMemoryAgent] message  {patient_id }")
        scrna_meta = {}
        scrna_msg = specialist_reports.get("OncologyAgent")
        if scrna_msg:
            scrna_meta = scrna_msg.metadata or {}
        cluster = (
            scrna_meta.get("effective_cluster") or scrna_meta.get("predicted_cluster") or "N/A"
        )
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_08", locals())
        user_content = render_ultra_prompt("ultra_user_content_prompt_14", locals())
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        record_data: Dict[str, Any] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "cluster": cluster,
            "query_snippet": query[:200],
        }
        try:
            resp = self._chat_completion(messages, model_name=model_name, temperature=0.1)
            raw = (resp.choices[0].message.content or "").strip()
            cleaned = re.sub(r"```json|```", "", raw).strip()
            extracted = json.loads(cleaned)
            record_data.update(extracted)
        except Exception as e:
            self._log(f"[LongitudinalMemoryAgent] LLMmessage {e } message")
            record_data["treatment_summary"] = "Status update."
        mem_path = self._get_memory_path(patient_id)
        existing: List[Dict] = []
        if mem_path.exists():
            try:
                existing = json.loads(mem_path.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing.append(record_data)
        try:
            mem_path.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._log(f"[LongitudinalMemoryAgent] message {mem_path }")
        except Exception as e:
            self._log(f"[LongitudinalMemoryAgent] message {e }")


class CounterfactualAgent(SpecialistAgent):
    AGENT_NAME = "CounterfactualAgent"
    ROLE_DESC = "Status update."
    SYSTEM_PROMPT_TEMPLATE = SPECIALIST_ROLE_PROMPTS["counterfactual"]

    def generate_report(
        self,
        query: str,
        tool_outputs: Dict[str, ToolResult],
        evidence_text: str,
        citations: List[Citation],
        model_name: Optional[str] = None,
        surgeon_report: str = "",
    ) -> AgentMessage:
        self._log_step(f"[{self .AGENT_NAME }] message")
        ref_text = self._build_citation_index(citations)
        user_content = render_ultra_prompt("ultra_user_content_prompt_15", locals())
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        resp = self._chat_completion(messages, model_name=model_name, temperature=0.3)
        content = (resp.choices[0].message.content or "").strip()
        self._log(f"[{self .AGENT_NAME }] message {len (content )} message ")
        return AgentMessage(
            agent_name=self.AGENT_NAME,
            role="specialist_report",
            content=content,
            citations=citations,
        )


class EvidenceGraderAgent(LLMCallerMixin):
    AGENT_NAME = "EvidenceGraderAgent"

    def __init__(
        self,
        model: str,
        trace_logs: List[str],
        step_counter_ref: List[int],
        stream_callback: Any = None,
    ) -> None:
        self.model = model
        self._trace_logs = trace_logs
        self._step_counter_ref = step_counter_ref
        self._stream_callback = stream_callback

    @property
    def _step_counter(self) -> int:
        return self._step_counter_ref[0]

    @_step_counter.setter
    def _step_counter(self, v: int) -> None:
        self._step_counter_ref[0] = v

    def grade(
        self,
        citations: List[Citation],
        synthesis_text: str,
        model_name: Optional[str] = None,
    ) -> Tuple[Dict[str, str], str]:
        _EXCLUDED_SOURCES = {
            "llm_mri",
            "raspr_run",
            "gigatime_infer",
            "llm_gigatime",
            "roam_infer",
            "scrna_pipeline_run",
        }
        literature_citations = [
            c
            for c in citations
            if (c.get("source") or "").lower() not in _EXCLUDED_SOURCES
            and not (c.get("title") or "").startswith("Tool output:")
        ]
        self._log_step(
            f"[EvidenceGraderAgent] message GRADE message"
            f" message {len (citations )} message message {len (literature_citations )} message "
        )
        if not literature_citations:
            return {}, ""
        citations = literature_citations
        grade_levels_desc = "\n".join([f" {k }: {v }" for k, v in GRADE_LEVELS.items()])
        cite_list = "\n".join(
            [
                f"[{i +1 }] {c .get ('title',"Status update.")} | Source={c .get ('source','N/A')} | PMID={c .get ('pmid','N/A')}"
                for i, c in enumerate(citations[:30])
            ]
        )
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_09", locals())
        user_content = render_ultra_prompt("ultra_user_content_prompt_16", locals())
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        citation_grades: Dict[str, str] = {}
        try:
            resp = self._chat_completion(messages, model_name=model_name, temperature=0.1)
            raw = (resp.choices[0].message.content or "").strip()
            cleaned = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(cleaned)
            citation_grades = data.get("grades", {})
        except Exception as e:
            self._log(f"[EvidenceGraderAgent] message {e } message")
            return {}, ""
        grade_counts: Dict[str, int] = {}
        for g in citation_grades.values():
            grade_counts[g] = grade_counts.get(g, 0) + 1
        summary_lines = [
            "\n---",
            "Status update.",
            "",
            "Status update.",
            "|------|------|-----------|",
        ]
        for lvl, desc in GRADE_LEVELS.items():
            cnt = grade_counts.get(lvl, 0)
            if cnt > 0:
                summary_lines.append(f"| {lvl } | {desc } | {cnt } |")
        summary_lines.append("")
        summary_lines.append("Status update.")
        for i, c in enumerate(citations[:30]):
            title_key = (c.get("title") or "")[:30]
            grade = citation_grades.get(title_key, "I")
            summary_lines.append(f"[{i +1 }] [{grade }] {c .get ('title',"Status update.")[:60 ]}")
        graded_text = "\n".join(summary_lines)
        self._log(f"[EvidenceGraderAgent] message {grade_counts }")
        return citation_grades, graded_text


class UncertaintyAgent(LLMCallerMixin):
    AGENT_NAME = "UncertaintyAgent"

    def __init__(
        self,
        model: str,
        trace_logs: List[str],
        step_counter_ref: List[int],
        stream_callback: Any = None,
    ) -> None:
        self.model = model
        self._trace_logs = trace_logs
        self._step_counter_ref = step_counter_ref
        self._stream_callback = stream_callback

    @property
    def _step_counter(self) -> int:
        return self._step_counter_ref[0]

    @_step_counter.setter
    def _step_counter(self, v: int) -> None:
        self._step_counter_ref[0] = v

    def analyze(
        self,
        synthesis_text: str,
        specialist_reports: Dict[str, "AgentMessage"],
        discussion_bus: "SharedBus",
        model_name: Optional[str] = None,
    ) -> str:
        self._log_step(f"[UncertaintyAgent] message")
        report_snippets = "\n\n".join(
            [f" {name } {msg .content [:600 ]}" for name, msg in specialist_reports.items()]
        )
        discussion_snippet = discussion_bus.get_full_thread_text()[:2000]
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_10", locals())
        user_content = render_ultra_prompt("ultra_user_content_prompt_17", locals())
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        uncertainty_data: Dict[str, Any] = {}
        try:
            resp = self._chat_completion(messages, model_name=model_name, temperature=0.1)
            raw = (resp.choices[0].message.content or "").strip()
            cleaned = re.sub(r"```json|```", "", raw).strip()
            uncertainty_data = json.loads(cleaned)
        except Exception as e:
            self._log(f"[UncertaintyAgent] message {e } ")
            return ""
        conclusions = uncertainty_data.get("key_conclusions") or []
        disagreements = uncertainty_data.get("major_disagreements") or []
        further_tests = uncertainty_data.get("recommend_further_tests") or []
        overall = uncertainty_data.get("overall_confidence", "N/A")
        overall_note = uncertainty_data.get("overall_note", "")
        conf_emoji = {"Status update.": "", "Status update.": "", "Status update.": ""}
        map_lines = [
            "\n---",
            "Status update.",
            f"message {conf_emoji .get (overall ,'')} {overall } {overall_note }",
            "",
            "Status update.",
            "|----------|--------|-------------|",
        ]
        for c in conclusions[:10]:
            conf = c.get("confidence", "Status update.")
            emoji = conf_emoji.get(conf, "")
            concl = (c.get("conclusion") or "")[:40]
            src = (c.get("uncertainty_source") or "")[:40]
            map_lines.append(f"| {concl } | {emoji } {conf } | {src } |")
        if disagreements:
            map_lines.append("")
            map_lines.append("Status update.")
            for d in disagreements:
                map_lines.append(f"- {d }")
        if further_tests:
            map_lines.append("")
            map_lines.append("Status update.")
            for t in further_tests:
                map_lines.append(f"- {t }")
        map_lines.append("")
        map_lines.append("Status update." "Status update." "Status update.")
        result_text = "\n".join(map_lines)
        self._log(f"[UncertaintyAgent] message message={overall }")
        return result_text


class PatientAdvocateAgent(LLMCallerMixin):
    AGENT_NAME = "PatientAdvocateAgent"

    def __init__(
        self,
        model: str,
        trace_logs: List[str],
        step_counter_ref: List[int],
        stream_callback: Any = None,
    ) -> None:
        self.model = model
        self._trace_logs = trace_logs
        self._step_counter_ref = step_counter_ref
        self._stream_callback = stream_callback

    @property
    def _step_counter(self) -> int:
        return self._step_counter_ref[0]

    @_step_counter.setter
    def _step_counter(self, v: int) -> None:
        self._step_counter_ref[0] = v

    def generate_patient_version(
        self,
        synthesis_text: str,
        query: str,
        numbered_questions=None,
        model_name=None,
    ) -> str:
        self._log_step("Status update.")
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_11", locals())
        user_content = render_ultra_prompt("ultra_user_content_prompt_18", locals())
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            resp = self._chat_completion(messages, model_name=model_name, temperature=0.35)
            patient_text = (resp.choices[0].message.content or "").strip()
            self._log(f"[PatientAdvocateAgent] message {len (patient_text )} message ")
            return "\n\n---\n" + patient_text
        except Exception as e:
            self._log(f"[PatientAdvocateAgent] message {e }")
            return ""


class LiteratureWatchAgent(LLMCallerMixin):
    AGENT_NAME = "LiteratureWatchAgent"

    def __init__(
        self,
        model: str,
        trace_logs: List[str],
        step_counter_ref: List[int],
        stream_callback: Any = None,
        watch_queue_file: Optional[Path] = None,
    ) -> None:
        self.model = model
        self._trace_logs = trace_logs
        self._step_counter_ref = step_counter_ref
        self._stream_callback = stream_callback
        self.watch_queue_file = watch_queue_file or WATCH_QUEUE_FILE
        self.watch_queue_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def _step_counter(self) -> int:
        return self._step_counter_ref[0]

    @_step_counter.setter
    def _step_counter(self, v: int) -> None:
        self._step_counter_ref[0] = v

    def register_keywords(
        self,
        query: str,
        synthesis_text: str,
        specialist_reports: Dict[str, "AgentMessage"],
        patient_id: str = "unknown",
        model_name: Optional[str] = None,
    ) -> None:
        self._log_step(f"[LiteratureWatchAgent] message")
        sys_prompt = render_ultra_prompt("ultra_sys_prompt_12", locals())
        user_content = render_ultra_prompt("ultra_user_content_prompt_19", locals())
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        keywords: List[str] = []
        pubmed_queries: List[str] = []
        try:
            resp = self._chat_completion(messages, model_name=model_name, temperature=0.1)
            raw = (resp.choices[0].message.content or "").strip()
            cleaned = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(cleaned)
            keywords = data.get("keywords", [])
            pubmed_queries = data.get("pubmed_queries", [])
        except Exception as e:
            self._log(f"[LiteratureWatchAgent] message {e } message")
            for word in re.findall(r"\b[A-Z]{3,}\b", query + " " + synthesis_text[:500]):
                if word not in keywords and len(keywords) < 5:
                    keywords.append(word)
        queue: List[Dict] = []
        if self.watch_queue_file.exists():
            try:
                queue = json.loads(self.watch_queue_file.read_text(encoding="utf-8"))
            except Exception:
                queue = []
        entry = {
            "patient_id": patient_id,
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "keywords": keywords,
            "pubmed_queries": pubmed_queries,
            "last_checked": None,
            "new_papers_found": 0,
            "status": "active",
        }
        queue.append(entry)
        try:
            self.watch_queue_file.write_text(
                json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._log(f"[LiteratureWatchAgent] message {keywords }")
        except Exception as e:
            self._log(f"[LiteratureWatchAgent] message {e }")

    def check_new_literature(self, max_entries: int = 5) -> List[Dict[str, Any]]:
        self._log(f"[LiteratureWatchAgent] message")
        if not self.watch_queue_file.exists():
            return []
        try:
            queue: List[Dict] = json.loads(self.watch_queue_file.read_text(encoding="utf-8"))
        except Exception as e:
            self._log(f"[LiteratureWatchAgent] message {e }")
            return []
        import requests

        alerts: List[Dict[str, Any]] = []
        updated = False
        for entry in queue[:max_entries]:
            if entry.get("status") != "active":
                continue
            queries = entry.get("pubmed_queries") or []
            if not queries:
                continue
            new_pmids: List[str] = []
            for q in queries[:2]:
                try:
                    r = requests.get(
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                        params={
                            "db": "pubmed",
                            "term": q,
                            "retmode": "json",
                            "retmax": 3,
                            "sort": "pub+date",
                        },
                        timeout=10,
                    )
                    ids = r.json().get("esearchresult", {}).get("idlist", [])
                    new_pmids.extend(ids)
                except Exception:
                    pass
            if new_pmids:
                entry["new_papers_found"] = len(new_pmids)
                entry["last_checked"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                alerts.append(
                    {
                        "patient_id": entry.get("patient_id"),
                        "keywords": entry.get("keywords"),
                        "new_pmids": new_pmids,
                    }
                )
                updated = True
        if updated:
            try:
                self.watch_queue_file.write_text(
                    json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
        return alerts


class OrchestratorAgent(LLMCallerMixin):
    def __init__(self, model: Optional[str] = None, backend: Optional[str] = None) -> None:
        self.model: str = model or GPT_MODEL
        self.backend: str = backend or LLM_BACKEND
        self.tools_schema = TOOLS_SCHEMA
        self._trace_logs: List[str] = []
        self._stream_callback = None
        self._step_counter_ref: List[int] = [0]
        self._run_id: Optional[str] = None
        self._run_dir: Optional[Path] = None
        self._tool_counts: Dict[str, int] = {}
        self._tool_events: List[Dict[str, Any]] = []
        self._tool_last_summary: Dict[str, str] = {}

    @property
    def _step_counter(self) -> int:
        return self._step_counter_ref[0]

    @_step_counter.setter
    def _step_counter(self, v: int) -> None:
        self._step_counter_ref[0] = v

    def _get_tool_desc(self) -> Dict[str, str]:
        d: Dict[str, str] = {}
        for t in self.tools_schema:
            fn = t.get("function") or {}
            name = fn.get("name")
            desc = fn.get("description") or ""
            if name:
                d[name] = desc
        return d

    def get_trace(self) -> str:
        return "\n".join(self._trace_logs)

    def _get_run_logs_root(self) -> Path:
        return Path("run_logs")

    def _init_run_artifacts(self, query: str) -> None:
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        root = self._get_run_logs_root()
        run_dir = root / self._run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        self._tool_counts = {}
        self._tool_events = []
        self._tool_last_summary = {}
        try:
            (run_dir / "input_query.txt").write_text(query or "", encoding="utf-8")
        except Exception:
            pass

    def _append_tool_usage_summary_csv(self, pipeline_agent: Optional[PipelineAgent]) -> None:
        if not self._run_dir or not self._run_id:
            return
        if pipeline_agent:
            for k, v in pipeline_agent._tool_counts.items():
                self._tool_counts[k] = self._tool_counts.get(k, 0) + v
            for k, v in pipeline_agent._tool_last_summary.items():
                self._tool_last_summary[k] = v
            self._tool_events.extend(pipeline_agent._tool_events)
        root = self._get_run_logs_root()
        root.mkdir(parents=True, exist_ok=True)
        summary_csv = root / "tool_usage_summary.csv"
        global_fieldnames = ["run_id", "tool_name", "result_summary", "count", "run_time"]
        per_run_fieldnames = ["tool_name", "count", "result_summary"]
        rows: List[Dict[str, Any]] = []
        run_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for tool_name, cnt in sorted(self._tool_counts.items()):
            summary = self._tool_last_summary.get(tool_name, "") or ""
            rows.append(
                {
                    "run_id": self._run_id,
                    "tool_name": tool_name,
                    "result_summary": summary,
                    "count": int(cnt),
                    "run_time": run_time_str,
                }
            )
        try:
            write_header = not summary_csv.exists()
            with summary_csv.open("a", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=global_fieldnames)
                if write_header:
                    w.writeheader()
                for r in rows:
                    w.writerow(r)
        except Exception:
            pass
        try:
            per_run_csv = self._run_dir / "tool_usage_this_run.csv"
            with per_run_csv.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=per_run_fieldnames)
                w.writeheader()
                for tool_name, cnt in sorted(self._tool_counts.items()):
                    w.writerow(
                        {
                            "tool_name": tool_name,
                            "count": int(cnt),
                            "result_summary": self._tool_last_summary.get(tool_name, "") or "",
                        }
                    )
        except Exception:
            pass

    def _dump_run_artifacts(
        self, final_answer: str, pipeline_agent: Optional[PipelineAgent] = None
    ) -> None:
        if not self._run_dir:
            return
        try:
            (self._run_dir / "trace.log").write_text(self.get_trace(), encoding="utf-8")
        except Exception:
            pass
        all_events = list(self._tool_events)
        if pipeline_agent:
            all_events.extend(pipeline_agent._tool_events)
        try:
            jsonl = self._run_dir / "tool_events.jsonl"
            with jsonl.open("w", encoding="utf-8") as f:
                for ev in all_events:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except Exception:
            pass
        try:
            (self._run_dir / "final_answer.txt").write_text(final_answer or "", encoding="utf-8")
        except Exception:
            pass
        self._append_tool_usage_summary_csv(pipeline_agent)

    def _dedupe_citations(self, citations: List[Citation]) -> List[Citation]:
        seen = set()
        uniq: List[Citation] = []
        for c in citations:
            key = (c.get("pmid"), c.get("title"), c.get("url"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        return uniq

    def _build_reference_section(self, citations: List[Citation]) -> str:
        uniq = self._dedupe_citations(citations)
        if not uniq:
            return ""
        lines = ["---", "Status update.", ""]
        for i, c in enumerate(uniq, start=1):
            meta = []
            if c.get("source"):
                meta.append(f"Source: {c ['source']}")
            if c.get("year"):
                meta.append(f"Year: {c ['year']}")
            if c.get("pmid"):
                meta.append(f"PMID: {c ['pmid']}")
            line = f"[{i }] {c ['title']}"
            if meta:
                line += f" ({'; '.join (meta )})"
            lines.append(line)
            if c.get("url"):
                lines.append(f"  {c ['url']}")
            lines.append("")
        return "\n".join(lines)

    def _suggest_tools_from_query(self, query: str, parsed: Dict[str, Any]) -> Plan:
        q = (query or "").strip()
        ql = q.lower()
        available = [
            t["function"]["name"] for t in self.tools_schema if t.get("type") == "function"
        ]
        suggested: List[str] = []
        scrna_markers = [
            "scrna",
            "sc-rna",
            "single cell",
            "single-cell",
            "Status update.",
            "scmulan",
            "cellchat",
            "h5ad",
            "10x",
            "10xgenomics",
            "matrix.mtx",
            "barcodes.tsv",
            "genes.tsv",
            "pseudotime",
            "seurat",
            "scanpy",
            "celltype",
            "cell type",
            "cluster",
            "patient_level_clustering",
            "Status update.",
            "Status update.",
            "Status update.",
            "patient classifier",
        ]
        has_h5ad = ".h5ad" in ql
        has_10x_like = any(x in ql for x in ["matrix.mtx", "barcodes.tsv", "genes.tsv", "10x"])
        is_scrna_intent = has_h5ad or has_10x_like or any(m in ql for m in scrna_markers)
        if is_scrna_intent and ("scrna_pipeline_run" in available):
            suggested.append("scrna_pipeline_run")
        img_suffixes = [".dcm", ".nii", ".nii.gz", ".png", ".jpg", ".jpeg", ".webp", ".bmp"]
        has_path = ("\\" in q) or ("/" in q)
        if any(s in ql for s in img_suffixes) or any(
            k in q
            for k in [
                "MRI",
                "mri",
                "Status update.",
                "Status update.",
                "Status update.",
                "FLAIR",
                "DWI",
            ]
        ):
            if "llm_mri" in available:
                suggested.append("llm_mri")
        is_folder_input = (
            ("folder" in ql)
            or ("Status update." in ql)
            or (has_path and not any(q.endswith(s) for s in [".png", ".jpg", ".jpeg"]))
        )
        is_survival_intent = any(
            k in ql
            for k in [
                "raspr",
                "run_case_all",
                "survival",
                "prognosis",
                "Status update.",
                "Status update.",
            ]
        )
        if (is_survival_intent or is_folder_input) and "raspr_run" in available:
            suggested.append("raspr_run")
        tiff_suffixes = [".tif", ".tiff"]
        if any(s in ql for s in tiff_suffixes) or any(
            k.lower() in ql
            for k in ["h&e", "he", "tiff", "Status update.", "Status update.", "Status update."]
        ):
            if "gigatime_infer" in available:
                suggested.append("gigatime_infer")
            if "roam_infer" in available:
                suggested.append("roam_infer")
        if "search_pubmed" in available:
            suggested.append("search_pubmed")
        if "search_google" in available:
            suggested.append("search_google")
        if "search_local_guidelines" in available:
            suggested.append("search_local_guidelines")
        if "search_local_facts" in available:
            suggested.append("search_local_facts")
        seen = set()
        suggested_unique: List[str] = []
        for s in suggested:
            if s not in seen:
                suggested_unique.append(s)
                seen.add(s)
        priority_tools = [
            "scrna_pipeline_run",
            "llm_mri",
            "raspr_run",
            "gigatime_infer",
            "roam_infer",
        ]
        final_order: List[str] = []
        for pt in priority_tools:
            if pt in suggested_unique:
                final_order.append(pt)
        for s in suggested_unique:
            if s not in priority_tools:
                final_order.append(s)
        notes = "Status update." "Status update." "Status update." "Status update." "Status update."
        return Plan(parsed=parsed, suggested_tools=final_order, notes=notes)

    def _analyze_patient_and_print_plan(self, query: str) -> Plan:
        self._log_step("Status update.")
        parsed = parse_question(query)
        self._log("Status update.")
        self._log(json.dumps(parsed, ensure_ascii=False, indent=2))
        self._log_step("Status update.")
        available = [
            t["function"]["name"] for t in self.tools_schema if t.get("type") == "function"
        ]
        self._log("Status update." + ", ".join(available))
        self._log_step("Status update.")
        plan = self._suggest_tools_from_query(query, parsed)
        self._log("Status update." + " -> ".join(plan.suggested_tools))
        self._log(plan.notes)
        return plan

    def _create_specialist_agents(self) -> Dict[str, SpecialistAgent]:
        kwargs = dict(
            model=self.model,
            trace_logs=self._trace_logs,
            step_counter_ref=self._step_counter_ref,
            stream_callback=self._stream_callback,
        )
        return {
            "PathologyAgent": PathologyAgent(**kwargs),
            "ImagingAgent": ImagingAgent(**kwargs),
            "OncologyAgent": OncologyAgent(**kwargs),
            "NeurosurgeryAgent": NeurosurgeryAgent(**kwargs),
            "RadiotherapyAgent": RadiotherapyAgent(**kwargs),
            "NursingAgent": NursingAgent(**kwargs),
        }

    def _run_specialist_reports(
        self,
        query: str,
        tool_outputs: Dict[str, ToolResult],
        evidence_text: str,
        citations: List[Citation],
        specialist_agents: Dict[str, SpecialistAgent],
        model_name: Optional[str] = None,
    ) -> Dict[str, AgentMessage]:
        self._log_step("Status update.")
        _ALL_OWNED_TOOLS: set = set()
        for agent in specialist_agents.values():
            owned = getattr(agent, "OWNED_TOOLS", [])
            _ALL_OWNED_TOOLS.update(owned)
        _SEARCH_TOOLS = {
            "search_local_guidelines",
            "search_local_facts",
            "search_pubmed",
            "search_google",
            "read_webpage",
        }

        def _route_tool_outputs(agent: SpecialistAgent) -> Dict[str, ToolResult]:
            owned = set(getattr(agent, "OWNED_TOOLS", []))
            routed: Dict[str, ToolResult] = {}
            for k, v in tool_outputs.items():
                if k in owned:
                    routed[k] = v
                elif k in _SEARCH_TOOLS:
                    routed[k] = v
                elif k not in _ALL_OWNED_TOOLS:
                    routed[k] = v
            return routed

        def _route_evidence_text(agent: SpecialistAgent) -> str:
            import re as _re

            owned = set(getattr(agent, "OWNED_TOOLS", []))
            _SEC_PAT = _re.compile(r"message")
            parts = _SEC_PAT.split(evidence_text)
            kept = [parts[0]]
            for i in range(1, len(parts) - 1, 2):
                tool_name = parts[i]
                body = parts[i + 1] if i + 1 < len(parts) else ""
                if tool_name in _ALL_OWNED_TOOLS and tool_name not in owned:
                    continue
                kept.append(f"=== {tool_name } message ==={body }")
            return "".join(kept)

        reports: Dict[str, AgentMessage] = {}
        for name, agent in specialist_agents.items():
            self._log(f" >> message {name }")
            routed_outputs = _route_tool_outputs(agent)
            owned_keys = [k for k in routed_outputs if k in set(getattr(agent, "OWNED_TOOLS", []))]
            routed_evidence = _route_evidence_text(agent)
            self._log(
                f"  {name } message {list (routed_outputs .keys ())} | "
                f"message {owned_keys } | evidence_textmessage {len (routed_evidence )}"
            )
            try:
                report = agent.generate_report(
                    query=query,
                    tool_outputs=routed_outputs,
                    evidence_text=routed_evidence,
                    citations=citations,
                    model_name=model_name,
                )
                reports[name] = report
                self._log(f" {name } message {len (report .content )} message ")
                print(f"\n{' '*60 }", flush=True)
                print(f" [{name }] message ", flush=True)
                print(f"{' '*60 }", flush=True)
                print(report.content, flush=True)
                print(f"{' '*60 }\n", flush=True)
            except Exception as e:
                self._log(f" {name } message {e }")
                reports[name] = AgentMessage(
                    agent_name=name,
                    role="specialist_report",
                    content=f" {name } message {e } ",
                    citations=[],
                )
        if self._run_dir:
            try:
                for _name, _rpt in reports.items():
                    safe_name = _name.replace("/", "_").replace("\\", "_")
                    (self._run_dir / f"specialist_report_{safe_name }.txt").write_text(
                        _rpt.content, encoding="utf-8"
                    )
            except Exception as _e:
                self._log(f"[Orchestrator] message {_e }")
        return reports

    def _run_discussion_rounds(
        self,
        query: str,
        specialist_reports: Dict[str, AgentMessage],
        specialist_agents: Dict[str, SpecialistAgent],
        auditor: "RiskAuditorAgent",
        model_name: Optional[str] = None,
        rounds: int = DISCUSSION_ROUNDS,
    ) -> SharedBus:
        self._log_step(
            f"[Orchestrator] message message {rounds } message message RiskAuditorAgent message "
        )
        bus = SharedBus()

        def _build_other_reports_summary(exclude: str) -> str:
            parts: List[str] = []
            for name, msg in specialist_reports.items():
                if name != exclude:
                    body = SpecialistAgent._strip_references_section(msg.content or "")
                    brief = body[:200] + "..." if len(body) > 200 else body
                    parts.append(f" {name } {brief }")
            return "\n\n".join(parts)

        for round_num in range(1, rounds + 1):
            self._log(f"\n{'='*50 }")
            self._log(f" [message] message {round_num } message")
            round_messages: List[AgentMessage] = []
            for agent_name, agent in specialist_agents.items():
                my_report = specialist_reports.get(agent_name)
                my_report_text = my_report.content if my_report else "Status update."
                other_reports = _build_other_reports_summary(agent_name)
                bus_thread = bus.get_thread_text(exclude_agent=agent_name)
                try:
                    msg = agent.generate_discussion_turn(
                        query=query,
                        my_report=my_report_text,
                        other_reports=other_reports,
                        bus_thread=bus_thread,
                        round_num=round_num,
                        model_name=model_name,
                    )
                    bus.post(msg)
                    round_messages.append(msg)
                    self._log(
                        f" [{agent_name }] message {round_num } message {len (msg .content )} message "
                    )
                    print(f"\n{' '*60 }", flush=True)
                    print(f" [{agent_name }] message {round_num } message ", flush=True)
                    print(f"{' '*60 }", flush=True)
                    print(msg.content, flush=True)
                    print(f"{' '*60 }\n", flush=True)
                except Exception as e:
                    self._log(f" [{agent_name }] message {round_num } message {e }")
                    err_msg = AgentMessage(
                        agent_name=agent_name,
                        role="discussion",
                        content=f" message {round_num } message {e } ",
                        metadata={"round": round_num},
                    )
                    bus.post(err_msg)
                    round_messages.append(err_msg)
            self._log(f" [message] message {round_num } message RiskAuditorAgent message...")
            try:
                audit_review = auditor.review_discussion_turn(
                    query=query,
                    round_num=round_num,
                    round_messages=round_messages,
                    model_name=model_name,
                )
                bus.post(audit_review)
                self._log(
                    f" [RiskAuditorAgent] message {round_num } message {len (audit_review .content )} message "
                )
                print(f"\n{' '*60 }", flush=True)
                print(f" [RiskAuditorAgent] message {round_num } message ", flush=True)
                print(f"{' '*60 }", flush=True)
                print(audit_review.content, flush=True)
                print(f"{' '*60 }\n", flush=True)
            except Exception as e:
                self._log(f" [RiskAuditorAgent] message {round_num } message {e }")
                bus.post(
                    AgentMessage(
                        agent_name="RiskAuditorAgent",
                        role="discussion_risk_review",
                        content=f" message {round_num } message {e } ",
                        metadata={"round": round_num, "type": "risk_review"},
                    )
                )
        if self._run_dir:
            try:
                (self._run_dir / "discussion_thread.txt").write_text(
                    bus.get_full_thread_text(), encoding="utf-8"
                )
            except Exception:
                pass
        self._log(f"[Orchestrator] message message {len (bus .messages )} message ")
        return bus

    def _run_synthesis_with_audit(
        self,
        query: str,
        specialist_reports: Dict[str, AgentMessage],
        discussion_bus: SharedBus,
        citations: List[Citation],
        numbered_questions: List[str],
        auditor: "RiskAuditorAgent",
        tool_outputs: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
    ) -> str:
        synthesizer = SynthesizerAgent(
            model=self.model,
            trace_logs=self._trace_logs,
            step_counter_ref=self._step_counter_ref,
            stream_callback=self._stream_callback,
        )
        audit_feedback: Optional[str] = None
        discussion_thread = discussion_bus.get_full_thread_text()
        for attempt in range(1, MAX_AUDIT_RETRIES + 2):
            draft = synthesizer.synthesize(
                query=query,
                specialist_reports=specialist_reports,
                discussion_bus=discussion_bus,
                citations=citations,
                numbered_questions=numbered_questions,
                audit_feedback=audit_feedback,
                tool_outputs=tool_outputs,
                model_name=model_name,
                attempt=attempt,
            )
            if attempt > MAX_AUDIT_RETRIES:
                self._log(f"[Orchestrator] message ({MAX_AUDIT_RETRIES }) message ")
                return draft
            passed, reason, audit_output = auditor.audit(
                draft_synthesis=draft,
                discussion_thread=discussion_thread,
                query=query,
                attempt=attempt,
                model_name=model_name,
            )
            if passed:
                self._log(f"[Orchestrator] message message {attempt } message {reason }")
                if "Status update." not in draft:
                    draft = (
                        draft
                        + f"\n\n message message{attempt }message | {datetime .now ().strftime ('%H:%M')} "
                    )
                return draft
            else:
                self._log(f"[Orchestrator] message message {attempt } message {reason }  message")
                _reason_lines = [
                    l
                    for l in audit_output.splitlines()
                    if l.startswith("REASON:") or l.startswith("C") and ". " in l[:5]
                ]
                audit_feedback = "\n".join(_reason_lines[:12]) or reason
        return draft

    def _run_with_gemini(self, query: str, model_name: str) -> str:
        if not GEMINI_ENABLED:
            return "Gemini disabled."
        self._log_step("Status update.")
        return gemini_generate_text(f"message: {query }")

    def _normalize_model_name(self, model_name: Optional[str]) -> str:
        if not model_name:
            return self.model
        return model_name

    def run(
        self,
        query: str,
        model_name: Optional[str] = None,
        stream_callback=None,
    ) -> str:
        normalized_model = self._normalize_model_name(model_name)
        start_time = time.time()
        self._trace_logs = []
        self._step_counter_ref = [0]
        self._stream_callback = stream_callback
        self._init_run_artifacts(query)
        self._log_step(
            f"[OrchestratorAgent] message GBM message v11.0 | Model: {normalized_model }"
        )
        static_plan = self._analyze_patient_and_print_plan(query)
        import sys as _sys

        _sys.stdout.flush()
        _sys.stderr.flush()
        print("Status update.", flush=True)
        _sys.stdout.flush()
        self._log_step("Status update.")
        planner = PlannerAgent(
            model=normalized_model,
            tools_schema=self.tools_schema,
            trace_logs=self._trace_logs,
            step_counter_ref=self._step_counter_ref,
            stream_callback=self._stream_callback,
        )
        plan = planner.plan(query=query, static_plan=static_plan, model_name=normalized_model)
        self._log_step("Status update.")
        pipeline_agent = PipelineAgent(
            model=normalized_model,
            tools_schema=self.tools_schema,
            trace_logs=self._trace_logs,
            step_counter_ref=self._step_counter_ref,
            stream_callback=self._stream_callback,
        )
        tool_outputs, citations, evidence_text = pipeline_agent.run(
            query=query,
            plan=plan,
            model_name=normalized_model,
        )
        self._log_step("Status update.")
        specialist_agents = self._create_specialist_agents()
        specialist_reports = self._run_specialist_reports(
            query=query,
            tool_outputs=tool_outputs,
            evidence_text=evidence_text,
            citations=citations,
            specialist_agents=specialist_agents,
            model_name=normalized_model,
        )
        self._log_step("Status update.")
        auditor = RiskAuditorAgent(
            model=normalized_model,
            trace_logs=self._trace_logs,
            step_counter_ref=self._step_counter_ref,
            stream_callback=self._stream_callback,
        )
        discussion_bus = self._run_discussion_rounds(
            query=query,
            specialist_reports=specialist_reports,
            specialist_agents=specialist_agents,
            auditor=auditor,
            model_name=normalized_model,
            rounds=DISCUSSION_ROUNDS,
        )
        if self._run_dir:
            try:
                (self._run_dir / "discussion_thread.txt").write_text(
                    discussion_bus.get_full_thread_text(), encoding="utf-8"
                )
            except Exception:
                pass
        self._log_step("Status update.")
        merged_citations: List[Citation] = list(citations)
        for sp_name, sp_msg in specialist_reports.items():
            sp_cites = sp_msg.citations or []
            merged_citations.extend(sp_cites)
            self._log(
                f" [{sp_name }] message {len (sp_cites )} message message {len (merged_citations )} message"
            )
        all_citations = self._dedupe_citations(merged_citations)
        self._log(f"[Orchestrator] message {len (all_citations )} message")
        self._log_step("Status update.")
        evidence_grader = EvidenceGraderAgent(
            model=normalized_model,
            trace_logs=self._trace_logs,
            step_counter_ref=self._step_counter_ref,
            stream_callback=self._stream_callback,
        )
        citation_grades, graded_text = evidence_grader.grade(
            citations=all_citations,
            synthesis_text="",
            model_name=normalized_model,
        )
        self._log_step("Status update.")
        numbered_questions = _extract_numbered_questions(query)
        final_synthesis = self._run_synthesis_with_audit(
            query=query,
            specialist_reports=specialist_reports,
            discussion_bus=discussion_bus,
            citations=all_citations,
            numbered_questions=numbered_questions,
            auditor=auditor,
            tool_outputs=tool_outputs,
            model_name=normalized_model,
        )
        self._log_step("Status update.")
        uncertainty_agent = UncertaintyAgent(
            model=normalized_model,
            trace_logs=self._trace_logs,
            step_counter_ref=self._step_counter_ref,
            stream_callback=self._stream_callback,
        )
        uncertainty_text = uncertainty_agent.analyze(
            synthesis_text=final_synthesis,
            specialist_reports=specialist_reports,
            discussion_bus=discussion_bus,
            model_name=normalized_model,
        )
        self._log_step("Status update.")
        patient_advocate = PatientAdvocateAgent(
            model=normalized_model,
            trace_logs=self._trace_logs,
            step_counter_ref=self._step_counter_ref,
            stream_callback=self._stream_callback,
        )
        patient_text = patient_advocate.generate_patient_version(
            synthesis_text=final_synthesis,
            query=query,
            model_name=normalized_model,
        )
        self._log_step("Status update.")
        lit_watch = LiteratureWatchAgent(
            model=normalized_model,
            trace_logs=self._trace_logs,
            step_counter_ref=self._step_counter_ref,
            stream_callback=self._stream_callback,
        )
        try:
            lit_watch.register_keywords(
                query=query,
                synthesis_text=final_synthesis,
                specialist_reports=specialist_reports,
                patient_id="current",
                model_name=normalized_model,
            )
        except Exception as e:
            self._log(f"[Orchestrator] LiteratureWatchAgent message message {e }")
        elapsed = time.time() - start_time
        ref_section = self._build_reference_section(all_citations)
        final_parts = [final_synthesis.strip() if final_synthesis else "Status update."]
        if uncertainty_text:
            final_parts.append(uncertainty_text)
        if graded_text:
            final_parts.append(graded_text)
        if ref_section:
            final_parts.append("\n" + ref_section)
        else:
            final_parts.append("Status update.")
        if patient_text:
            final_parts.append(patient_text)
        professional_source_parts = list(final_parts)
        if (
            patient_text
            and professional_source_parts
            and professional_source_parts[-1] == patient_text
        ):
            professional_source_parts = professional_source_parts[:-1]
        professional_answer = "\n".join(professional_source_parts).strip()
        patient_answer = (patient_text or "").strip()
        if patient_answer.startswith("---"):
            patient_answer = patient_answer[3:].strip()
        final_parts = [
            FINAL_SECTION_TITLES["mdt"],
            professional_answer,
            "",
            FINAL_SECTION_TITLES["patient"],
            patient_answer or PATIENT_FRIENDLY_FALLBACK,
            "",
            FINAL_SECTION_TITLES["professional"],
            professional_answer,
        ]
        agent_summary_lines = ["\n---", "Status update.", ""]
        agent_summary_lines.append(
            "Status update."
            "PathologyAgent | ImagingAgent | OncologyAgent | "
            "NeurosurgeryAgent | RadiotherapyAgent | NursingAgent | "
            "RiskAuditorAgent | EvidenceGraderAgent | UncertaintyAgent | "
            "SynthesizerAgent | PatientAdvocateAgent | LiteratureWatchAgent"
        )
        agent_summary_lines.append(
            f"- message message PathologyAgent | message ImagingAgent | scRNAmessage OncologyAgent"
        )
        agent_summary_lines.append(f"- message {len (tool_outputs )}")
        agent_summary_lines.append(
            f"- message {len (discussion_bus .messages )} | message {DISCUSSION_ROUNDS }"
        )
        agent_summary_lines.append(f"- message message {len (all_citations )}")
        agent_summary_lines.append(f"- message {elapsed :.1f} message")
        final_parts.append("\n".join(agent_summary_lines))
        final_parts.append(f"\n \n(message {elapsed :.1f} message)")
        final_answer = "\n".join(final_parts)
        self._dump_run_artifacts(final_answer, pipeline_agent=pipeline_agent)
        return final_answer


class UltraGBMAgent(OrchestratorAgent):
    pass


if __name__ == "__main__":
    agent = UltraGBMAgent()
    print(" CLI Mode  Ultra GBM Multi-Agent System")
    print("Status update.")
    print("   NeurosurgeryAgent | RadiotherapyAgent | NursingAgent | RiskAuditorAgent")
    print(
        "   EvidenceGraderAgent | UncertaintyAgent | SynthesizerAgent | PatientAdvocateAgent | LiteratureWatchAgent"
    )
    while True:
        try:
            q = input("Status update.")
            if not q:
                continue
            print(agent.run(q))
            print("\nTrace:\n" + agent.get_trace())
        except KeyboardInterrupt:
            print("Status update.")
            break
        except Exception as e:
            print(f"ERROR: {e }")
