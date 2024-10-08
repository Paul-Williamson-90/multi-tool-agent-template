import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.prompt_templates.router_template import SYSTEM_PROMPT

def test_system_prompt():
    assert isinstance(SYSTEM_PROMPT, str)