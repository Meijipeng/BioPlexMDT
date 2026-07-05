# Prompt Filling Guide

Prompts are filled in:

```text
src/gbm_agent/prompts.py
```

The public repository intentionally does not provide concrete medical, clinical, or agent-control prompts. Each placeholder states that researchers must fill the prompt according to their own study protocol, institutional requirements, safety policy, evidence standard, target language, and validation framework before running BioPlexMDT.

To add prompts, write the prompt text into the corresponding variable or function return value. Do not change variable names, function names, or dictionary keys.

## Prompt Locations

| Purpose | Location |
|---|---|
| Question parsing | `QUESTION_PARSER_SYSTEM_PROMPT`, `question_parser_user_prompt()` |
| Literature or trial fact extraction | `TRIAL_FACT_EXTRACTION_SYSTEM_PROMPT`, `trial_fact_user_prompt()` |
| MRI/DICOM image analysis | `MRI_VISION_SYSTEM_PROMPT`, `mri_vision_user_prompt()` |
| GigaTIME simulated immune-map analysis | `GIGATIME_VISION_SYSTEM_PROMPT`, `gigatime_vision_user_prompt()` |
| Case context construction | `build_input_context_prompt()` |
| MDT specialist roles | `SPECIALIST_BASE_PROMPT_TEMPLATE`, `SPECIALIST_INITIAL_ISOLATION_PROMPT`, `SPECIALIST_DISCUSSION_PROMPT`, `SPECIALIST_ROLE_PROMPTS` |
| Main workflow control | `pipeline_system_prompt()` |
| Final section titles | `FINAL_SECTION_TITLES` |
| Advanced workflow prompts | `ULTRA_PROMPT_EXPRESSIONS`, `ULTRA_USER_PROMPT_EXPRESSIONS` |

## How to Fill Prompts

For long prompts, use triple-quoted strings:

```python
QUESTION_PARSER_SYSTEM_PROMPT = """
Write the system prompt for question parsing here.
Ask the model to extract structured fields from the case description and return JSON.
"""
```

When the prompt needs to include input content, keep the function name and return a string inside the function:

```python
def question_parser_user_prompt(question: str) -> str:
    return f"""
Please parse the following case question:
{question}
"""
```

Specialist role prompts are filled in the dictionary:

```python
SPECIALIST_ROLE_PROMPTS = {
    "imaging": "Write the imaging specialist prompt here.",
    "pathology": "Write the pathology specialist prompt here.",
    "oncology": "Write the medical oncology specialist prompt here.",
    "neurosurgery": "Write the neurosurgery specialist prompt here.",
    "radiotherapy": "Write the radiotherapy specialist prompt here.",
    "nursing": "Write the nursing and follow-up prompt here.",
    "counterfactual": "Write the counterfactual review prompt here.",
}
```

Only replace the text inside quotation marks. Keep variable names, function names, and dictionary keys unchanged.
