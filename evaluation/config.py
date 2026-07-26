import os

API_KEY = os.environ.get("EMRB_API_KEY", "")
API_BASE_URL = os.environ.get("EMRB_API_BASE_URL", "https://api.openai.com/v1")

MODEL_PROVIDERS = {
    # OpenAI
    "gpt-4o": ("openai", API_KEY, API_BASE_URL),
    "gpt-4o-mini": ("openai", API_KEY, API_BASE_URL),
    "gpt-5": ("openai", API_KEY, API_BASE_URL),
    "gpt-5.5": ("openai", API_KEY, API_BASE_URL),
    # Anthropic
    "claude-sonnet-4-6": ("anthropic", API_KEY, API_BASE_URL),
    "claude-opus-4-6": ("anthropic", API_KEY, API_BASE_URL),
    "claude-opus-4-7": ("anthropic", API_KEY, API_BASE_URL),
    # DeepSeek
    "deepseek-chat": ("deepseek", API_KEY, API_BASE_URL),
    "deepseek-reasoner": ("deepseek", API_KEY, API_BASE_URL),
    "deepseek-v4-pro": ("deepseek", API_KEY, API_BASE_URL),
    "deepseek-v4-flash": ("deepseek", API_KEY, API_BASE_URL),
    # Google
    "gemini-3.1-pro-preview": ("gemini", API_KEY, API_BASE_URL),
    "gemini-3.1-flash-lite-preview": ("gemini", API_KEY, API_BASE_URL),
    "gemini-3.5-flash": ("gemini", API_KEY, API_BASE_URL),
    # Other
    "llama-3.3-70b": ("openai", API_KEY, API_BASE_URL),
    "glm-5.2": ("openai", API_KEY, API_BASE_URL),
    "kimi-k2.6": ("openai", API_KEY, API_BASE_URL),
    "minimax-m3": ("openai", API_KEY, API_BASE_URL),
}

DEFAULT_MODEL = "gpt-4o"
MAX_TURNS = 15
CODE_TIMEOUT = 60
MAX_OUTPUT_LEN = 15000


def get_client_config(model):
    if model in MODEL_PROVIDERS:
        _, api_key, base_url = MODEL_PROVIDERS[model]
        return api_key, base_url
    return API_KEY, API_BASE_URL
