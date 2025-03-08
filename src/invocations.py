import logging
import json
from typing import Any, Type

from ast import literal_eval
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed, before_log
from llama_index.core.llms.llm import LLM
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import CompletionResponse


logger = logging.getLogger(__name__)


PydanticType = Type[BaseModel]


DEFAULT_STRUCTURED_PROMPT_TEMPLATE = PromptTemplate(
    "{context}\n\n"
    "**Below is the schema for which your JSON output must be formatted as "
    "(YOU MUST ensure that you are double escaping any control characters that are not inside strings):**\n"
    "{schema}"
)


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1), before=before_log(logger, logging.INFO))
def structured_invocation(
    llm: LLM,
    context: str,
    pydantic_object: type[BaseModel],
    prompt_template: PromptTemplate = DEFAULT_STRUCTURED_PROMPT_TEMPLATE,
    llm_kwargs: dict[str, Any] = {},
) -> PydanticType:
    response = llm.complete(
        prompt=prompt_template.format(
            context=context,
            schema=json.dumps(pydantic_object.model_json_schema()),
        ),
        **llm_kwargs,
    )
    response_str = str(response.text).strip()
    response_json = literal_eval(response_str)
    response_object = pydantic_object(**response_json)
    return response_object


@retry(stop=stop_after_attempt(5), wait=wait_fixed(1), before=before_log(logger, logging.INFO))
def non_structured_invocation(
    llm: LLM,
    prompt: str,
    inference_kwargs: dict[str, Any] = {},
) -> str:
    response: CompletionResponse = llm.complete(
        prompt=prompt, **inference_kwargs
    )
    response_str = str(response)

    return response_str