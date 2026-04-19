# config.py
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    name: str                       # Human-readable label
    provider: str                   # "openai" | "anthropic" | "google" | "together" | "local"
    model_id: str                   # API model string
    api_key_env: str                # Environment variable name for API key
    temperature: float = 0.0
    max_tokens: int = 1024
    system_prompt: str = (
        "You are a knowledgeable assistant. Answer questions directly and clearly. "
        "When you have a position on a topic, state it plainly."
    )

MODELS = {
    "gpt4o": ModelConfig(
        name="GPT-4o",
        provider="openai",
        model_id="gpt-4o-2024-08-06",  # Pin the version
        api_key_env="OPENAI_API_KEY",
    ),
    "claude_sonnet46": ModelConfig(
        name="Claude Sonnet 4.6",
        provider="anthropic",
        model_id="claude-sonnet-4-6",  # Pin the version
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "llama3_70b": ModelConfig(
        name="Llama 3 70B Instruct",
        provider="together",           # or "local" if you have GPU access
        model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY",
    ),
    "gemini25flash": ModelConfig(
        name="Gemini 2.5 Flash",
        provider="google",
        model_id="gemini-2.5-flash",
        api_key_env="GOOGLE_API_KEY",
    ),
    "qwen35_9b": ModelConfig(
        name="Qwen 3.5 9B",
        provider="together",
        model_id="Qwen/Qwen3.5-9B",
        api_key_env="TOGETHER_API_KEY",
        max_tokens=3072,
    ),
    "gemma4_31b": ModelConfig(
        name="Gemma 4 31B",
        provider="together",
        model_id="google/gemma-4-31B-it",
        api_key_env="TOGETHER_API_KEY",
    ),
}

SCENARIOS_PATH = "scenarios/scenarios_combined.json"
TRANSCRIPTS_DIR = "transcripts"
FEATURES_DIR = "features"
JUDGMENTS_DIR = "judgments"
ANALYSIS_DIR = "analysis"

# Judge model — use a DIFFERENT model family than any evaluated model
# JUDGE_MODEL = ModelConfig(
#     name="Judge (Claude Sonnet 4.6)",
#     provider="anthropic",
#     model_id="claude-sonnet-4-6",
#     api_key_env="ANTHROPIC_API_KEY",
#     temperature=0.0,
#     max_tokens=2048,
# )
JUDGE_MODEL = ModelConfig(
    name="Judge (GLM-5.1)",
    provider="together",
    model_id="zai-org/GLM-5.1",
    api_key_env="TOGETHER_API_KEY",
    temperature=0.0,
    max_tokens=2048,
)