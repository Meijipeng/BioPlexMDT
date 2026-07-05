from __future__ import annotations

from typing import Iterable, Mapping

PROMPT_FILLING_NOTICE = (
    "PROMPT TEMPLATE NOT PROVIDED. This public repository intentionally does not "
    "provide concrete medical, clinical, or agent-control prompts. Researchers "
    "must fill this prompt according to their own study protocol, institutional "
    "requirements, safety policy, evidence standard, target language, and "
    "validation framework before running BioPlexMDT."
)


def _prompt_placeholder(name: str) -> str:
    return f"{name}: {PROMPT_FILLING_NOTICE}"


QUESTION_PARSER_SYSTEM_PROMPT = _prompt_placeholder("QUESTION_PARSER_SYSTEM_PROMPT")


def question_parser_user_prompt(question: str) -> str:
    return _prompt_placeholder("question_parser_user_prompt")


TRIAL_FACT_EXTRACTION_SYSTEM_PROMPT = _prompt_placeholder("TRIAL_FACT_EXTRACTION_SYSTEM_PROMPT")


def trial_fact_user_prompt(pmid: str, title: str, abstract: str) -> str:
    return _prompt_placeholder("trial_fact_user_prompt")


MRI_VISION_SYSTEM_PROMPT = _prompt_placeholder("MRI_VISION_SYSTEM_PROMPT")


def mri_vision_user_prompt(question: str) -> str:
    return _prompt_placeholder("mri_vision_user_prompt")


GIGATIME_VISION_SYSTEM_PROMPT = _prompt_placeholder("GIGATIME_VISION_SYSTEM_PROMPT")


def gigatime_vision_user_prompt(question: str) -> str:
    return _prompt_placeholder("gigatime_vision_user_prompt")


def build_input_context_prompt(meta: Mapping[str, str]) -> str:
    return _prompt_placeholder("build_input_context_prompt")


SPECIALIST_BASE_PROMPT_TEMPLATE = _prompt_placeholder("SPECIALIST_BASE_PROMPT_TEMPLATE")
SPECIALIST_INITIAL_ISOLATION_PROMPT = _prompt_placeholder(
    "SPECIALIST_INITIAL_ISOLATION_PROMPT"
)
SPECIALIST_DISCUSSION_PROMPT = _prompt_placeholder("SPECIALIST_DISCUSSION_PROMPT")
SPECIALIST_ROLE_PROMPTS = {
    "imaging": _prompt_placeholder("SPECIALIST_ROLE_PROMPTS.imaging"),
    "pathology": _prompt_placeholder("SPECIALIST_ROLE_PROMPTS.pathology"),
    "oncology": _prompt_placeholder("SPECIALIST_ROLE_PROMPTS.oncology"),
    "neurosurgery": _prompt_placeholder("SPECIALIST_ROLE_PROMPTS.neurosurgery"),
    "radiotherapy": _prompt_placeholder("SPECIALIST_ROLE_PROMPTS.radiotherapy"),
    "nursing": _prompt_placeholder("SPECIALIST_ROLE_PROMPTS.nursing"),
    "counterfactual": _prompt_placeholder("SPECIALIST_ROLE_PROMPTS.counterfactual"),
}


def specialist_system_prompt(role: str) -> str:
    return _prompt_placeholder("specialist_system_prompt")


def pipeline_system_prompt(
    required_tools: Iterable[str], plan_hint: str, numbered_questions_text: str
) -> str:
    return _prompt_placeholder("pipeline_system_prompt")


FINAL_SECTION_TITLES = {
    "mdt": "## MDT Recommendation",
    "patient": "## Patient-Friendly Explanation",
    "professional": "## Professional Answer",
}
PATIENT_FRIENDLY_FALLBACK = _prompt_placeholder("PATIENT_FRIENDLY_FALLBACK")
ULTRA_PROMPT_EXPRESSIONS = {
    "ultra_sys_prompt_01": _prompt_placeholder("ultra_sys_prompt_01"),
    "ultra_sys_prompt_02": _prompt_placeholder("ultra_sys_prompt_02"),
    "ultra_sys_prompt_03": _prompt_placeholder("ultra_sys_prompt_03"),
    "ultra_sys_prompt_04": _prompt_placeholder("ultra_sys_prompt_04"),
    "ultra_sys_prompt_05": _prompt_placeholder("ultra_sys_prompt_05"),
    "ultra_sys_prompt_06": _prompt_placeholder("ultra_sys_prompt_06"),
    "ultra_sys_prompt_07": _prompt_placeholder("ultra_sys_prompt_07"),
    "ultra_sys_prompt_08": _prompt_placeholder("ultra_sys_prompt_08"),
    "ultra_sys_prompt_09": _prompt_placeholder("ultra_sys_prompt_09"),
    "ultra_sys_prompt_10": _prompt_placeholder("ultra_sys_prompt_10"),
    "ultra_sys_prompt_11": _prompt_placeholder("ultra_sys_prompt_11"),
    "ultra_sys_prompt_12": _prompt_placeholder("ultra_sys_prompt_12"),
}
ULTRA_USER_PROMPT_EXPRESSIONS = {
    "ultra_prompt_prompt_01": _prompt_placeholder("ultra_prompt_prompt_01"),
    "ultra_user_content_prompt_02": _prompt_placeholder("ultra_user_content_prompt_02"),
    "ultra_user_content_prompt_03": _prompt_placeholder("ultra_user_content_prompt_03"),
    "ultra_user_content_prompt_04": _prompt_placeholder("ultra_user_content_prompt_04"),
    "ultra_user_content_prompt_05": _prompt_placeholder("ultra_user_content_prompt_05"),
    "ultra_user_content_prompt_06": _prompt_placeholder("ultra_user_content_prompt_06"),
    "ultra_user_content_prompt_07": _prompt_placeholder("ultra_user_content_prompt_07"),
    "ultra_user_content_prompt_08": _prompt_placeholder("ultra_user_content_prompt_08"),
    "ultra_user_content_prompt_09": _prompt_placeholder("ultra_user_content_prompt_09"),
    "ultra_user_content_prompt_10": _prompt_placeholder("ultra_user_content_prompt_10"),
    "ultra_user_content_prompt_11": _prompt_placeholder("ultra_user_content_prompt_11"),
    "ultra_user_content_prompt_12": _prompt_placeholder("ultra_user_content_prompt_12"),
    "ultra_user_content_prompt_13": _prompt_placeholder("ultra_user_content_prompt_13"),
    "ultra_user_content_prompt_14": _prompt_placeholder("ultra_user_content_prompt_14"),
    "ultra_user_content_prompt_15": _prompt_placeholder("ultra_user_content_prompt_15"),
    "ultra_user_content_prompt_16": _prompt_placeholder("ultra_user_content_prompt_16"),
    "ultra_user_content_prompt_17": _prompt_placeholder("ultra_user_content_prompt_17"),
    "ultra_user_content_prompt_18": _prompt_placeholder("ultra_user_content_prompt_18"),
    "ultra_user_content_prompt_19": _prompt_placeholder("ultra_user_content_prompt_19"),
}


def render_ultra_prompt(name: str, scope: Mapping[str, object]) -> str:
    return _prompt_placeholder(name)
