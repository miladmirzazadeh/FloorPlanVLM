"""The ONE static system prompt, used identically across every training sample.

Phrasing is frozen on purpose: changing punctuation/wording between samples injects
token variance that distracts the model from learning the coordinate patterns. The
schema text comes from schema.schema_doc() so the prompt and the encoded targets can
never disagree.
"""
from . import schema

SYSTEM_PROMPT = (
    "You are a floor plan vectorization expert. Extract the building's wall geometry "
    "from the image as structured JSON.\n\n"
    + schema.schema_doc()
)

USER_PROMPT = "Extract the walls from this floor plan."
