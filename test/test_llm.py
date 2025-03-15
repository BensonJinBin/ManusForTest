from http.client import responses

import pytest

from app.config import LLMSettings
from pydantic import ValidationError

from app.llm import LLM


def test_llm_settings_valid_input():
    # 测试使用有效的输入创建 LLMSettings 实例
    settings = LLMSettings(
        model="gpt-3.5-turbo",
        base_url="https://api.example.com",
        api_key="your_api_key",
        max_tokens=2048,
        temperature=0.7,
        api_type="Openai",
        api_version="v1"
    )
    assert settings.model == "gpt-3.5-turbo"
    assert settings.base_url == "https://api.example.com"
    assert settings.api_key == "your_api_key"
    assert settings.max_tokens == 2048
    assert settings.temperature == 0.7
    assert settings.api_type == "Openai"
    assert settings.api_version == "v1"

def test_llm_settings_missing_required_fields():
    # 测试缺少必需字段时是否抛出 ValidationError
    try:
        LLMSettings(
            base_url="https://api.example.com",
            api_key="your_api_key",
            max_tokens=2048,
            temperature=0.7,
            api_type="Openai",
            api_version="v1"
        )
    except ValidationError as e:
        assert "model" in str(e)

def test_llm_settings_default_values():
    # 测试使用默认值创建 LLMSettings 实例
    settings = LLMSettings(
        model="gpt-3.5-turbo",
        base_url="https://api.example.com",
        api_key="your_api_key",
        api_type="Openai",
        api_version="v1"
    )
    assert settings.max_tokens == 4096
    assert settings.temperature == 1.0

@pytest.fixture
def llm_config():
    return LLMSettings(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-vudvzatxlxndutsxdzmpzoprhndmvjvuglxshuhiblkmbtxu",
        max_tokens=2048,
        temperature=0.5,
        api_type="openai",
        api_version="v1"
    )

@pytest.mark.asyncio
async def test_llm(llm_config):
    llm = LLM("Qwen2.5-7B-Instruct", llm_config)
    messages = [{"role": "user", "content": "introduce yourself"}]
    response = await llm.ask(messages)
    assert "Qwen" in response