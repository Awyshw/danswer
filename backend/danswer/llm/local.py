import os
from typing import Any

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models.openai import ChatOpenAI

from danswer.configs.model_configs import API_BASE_OPENAI
from danswer.configs.model_configs import API_MODEL_NAME
from danswer.llm.llm import LangChainChatLLM
from danswer.llm.utils import should_be_verbose


class LocalGPT(LangChainChatLLM):
    def __init__(
        self,
        api_key: str,
        max_output_tokens: int,
        timeout: int,
        model_version: str,
        api_base: str = API_BASE_OPENAI,
        streaming: bool = True,
        temperature: float = 0.7,
        *args: list[Any],
        **kwargs: dict[str, Any]
    ):
        # set a dummy API key if not specified so that LangChain doesn't throw an
        # exception when trying to initialize the LLM which would prevent the API
        # server from starting up
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
        callback = AsyncIteratorCallbackHandler()
        self._llm = ChatOpenAI(
            # streaming=streaming,
            callbacks=[callback],
            openai_api_key=api_key,
            openai_api_base=api_base,
            model_name=model_version,
            max_tokens=max_output_tokens,
            temperature=temperature,
            request_timeout=timeout,
            model_kwargs={
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            },
            verbose=should_be_verbose(),
            max_retries=0,  # retries are handled outside of langchain
        )

    @property
    def llm(self) -> ChatOpenAI:
        return self._llm
