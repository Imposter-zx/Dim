# dim_ai.py — AI/LLM Engine for Dim
#
# Provides typed prompts, model adapters, and AI integration.

import os
import json
import asyncio
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    provider: ModelProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60


@dataclass
class PromptTemplate:
    name: str
    system_prompt: str
    user_template: str
    output_schema: Optional[Dict[str, Any]] = None


class ModelAdapter:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None

    def initialize(self) -> bool:
        if self.config.provider == ModelProvider.OPENAI:
            return self._init_openai()
        elif self.config.provider == ModelProvider.ANTHROPIC:
            return self._init_anthropic()
        elif self.config.provider == ModelProvider.LOCAL:
            return self._init_local()
        return False

    def _init_openai(self) -> bool:
        try:
            import openai

            self._client = openai.OpenAI(
                api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.config.base_url or "https://api.openai.com/v1",
            )
            return True
        except ImportError:
            return False

    def _init_anthropic(self) -> bool:
        try:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            return True
        except ImportError:
            return False

    def _init_local(self) -> bool:
        self._client = LocalModelClient(self.config.base_url or "http://localhost:8080")
        return True

    def generate(self, system_prompt: str, user_input: str, **kwargs) -> str:
        if not self._client:
            return self._stub_response(user_input)

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        if self.config.provider == ModelProvider.OPENAI:
            return self._generate_openai(
                system_prompt, user_input, temperature, max_tokens
            )
        elif self.config.provider == ModelProvider.ANTHROPIC:
            return self._generate_anthropic(
                system_prompt, user_input, temperature, max_tokens
            )
        else:
            return self._generate_local(system_prompt, user_input)

    def _generate_openai(
        self, system: str, user: str, temp: float, max_tok: int
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temp,
                max_tokens=max_tok,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def _generate_anthropic(
        self, system: str, user: str, temp: float, max_tok: int
    ) -> str:
        try:
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=max_tok,
                temperature=temp,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        except Exception as e:
            return f"Error: {e}"

    def _generate_local(self, system: str, user: str) -> str:
        try:
            import urllib.request
            import urllib.parse

            data = json.dumps(
                {
                    "model": self.config.model,
                    "prompt": f"System: {system}\nUser: {user}",
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
            )

            req = urllib.request.Request(
                self.config.base_url,
                data=data.encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "")
        except Exception as e:
            return f"Error: {e}"

    def _stub_response(self, input_str: str) -> str:
        return f"[AI Response to: {input_str[:50]}...]"


class LocalModelClient:
    def __init__(self, base_url: str):
        self.base_url = base_url


class AIEngine:
    def __init__(self):
        self.adapters: Dict[str, ModelAdapter] = {}
        self.prompts: Dict[str, PromptTemplate] = {}
        self.default_adapter: Optional[str] = None

    def register_adapter(self, name: str, adapter: ModelAdapter) -> bool:
        if adapter.initialize():
            self.adapters[name] = adapter
            if self.default_adapter is None:
                self.default_adapter = name
            return True
        return False

    def register_prompt(self, template: PromptTemplate):
        self.prompts[template.name] = template

    def create_prompt(self, name: str, **kwargs) -> str:
        if name not in self.prompts:
            return f"Error: Prompt '{name}' not found"

        template = self.prompts[name]
        user_input = template.user_template.format(**kwargs)
        return user_input

    def execute_prompt(self, name: str, **kwargs) -> str:
        if name not in self.prompts:
            return f"Error: Prompt '{name}' not found"

        template = self.prompts[name]
        user_input = template.user_template.format(**kwargs)

        adapter_name = self.default_adapter
        if adapter_name not in self.adapters:
            return "Error: No model adapter configured"

        adapter = self.adapters[adapter_name]
        return adapter.generate(template.system_prompt, user_input)

    def chat(
        self, messages: List[Dict[str, str]], adapter: Optional[str] = None
    ) -> str:
        adapter_name = adapter or self.default_adapter
        if adapter_name not in self.adapters:
            return "Error: No model adapter configured"

        adapter = self.adapters[adapter_name]
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")

        return adapter.generate(system, user)


def create_openai_adapter(model: str = "gpt-4", **kwargs) -> ModelAdapter:
    config = ModelConfig(provider=ModelProvider.OPENAI, model=model, **kwargs)
    return ModelAdapter(config)


def create_anthropic_adapter(model: str = "claude-3-opus", **kwargs) -> ModelAdapter:
    config = ModelConfig(provider=ModelProvider.ANTHROPIC, model=model, **kwargs)
    return ModelAdapter(config)


def create_local_adapter(
    model: str = "llama2", base_url: str = "http://localhost:8080", **kwargs
) -> ModelAdapter:
    config = ModelConfig(
        provider=ModelProvider.LOCAL, model=model, base_url=base_url, **kwargs
    )
    return ModelAdapter(config)


def create_prompt(
    name: str, system: str, user: str, output_schema: Optional[Dict] = None
) -> PromptTemplate:
    return PromptTemplate(
        name=name, system_prompt=system, user_template=user, output_schema=output_schema
    )


def run_ai_demo():
    engine = AIEngine()

    adapter = create_openai_adapter("gpt-4")
    if engine.register_adapter("openai", adapter):
        print("✓ OpenAI adapter registered")
    else:
        print("✗ OpenAI not available (install openai package)")
        adapter = create_local_adapter()
        if engine.register_adapter("local", adapter):
            print("✓ Local adapter registered")

    engine.register_prompt(
        create_prompt(
            name="classify",
            system="You are a text classifier. Output only one word: positive, negative, or neutral.",
            user_template="Classify: {text}",
        )
    )

    result = engine.execute_prompt("classify", text="I love this product!")
    print(f"Classification: {result}")


if __name__ == "__main__":
    run_ai_demo()
