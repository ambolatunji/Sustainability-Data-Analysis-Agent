from __future__ import annotations
import os
import requests
from typing import List, Dict, Optional

def _get_secret(name: str) -> Optional[str]:
    # 1) Streamlit secrets (supports nested under [api] or top-level)
    try:
        import streamlit as st
        if "api" in st.secrets and name in st.secrets["api"]:
            return st.secrets["api"][name]
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)

class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider.lower().strip()
        self.model = model
        self.timeout = 60

    def chat(self, messages: List[Dict], temperature: float = 0.2, max_tokens: int = 800) -> str:
        if self.provider == 'openai':
            key = os.getenv('OPENAI_API_KEY')
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            url = 'https://api.openai.com/v1/chat/completions'
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content']

        if self.provider == 'groq':
            key = os.getenv('GROQ_API_KEY')
            if not key:
                raise RuntimeError("GROQ_API_KEY not set")
            url = 'https://api.groq.com/openai/v1/chat/completions'
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content']

        if self.provider == 'deepseek':
            key = os.getenv('DEEPSEEK_API_KEY')
            if not key:
                raise RuntimeError("DEEPSEEK_API_KEY not set")
            base = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
            url = f'{base}/v1/chat/completions'
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content']

        raise ValueError(f"Unsupported provider: {self.provider}")
