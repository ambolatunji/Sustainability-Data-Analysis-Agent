from __future__ import annotations
import importlib.resources as pkg_resources

# Tiny loader that reads *.txt files as strings at import time

def _read(name: str) -> str:
    with pkg_resources.files(__package__).joinpath(name).open('r', encoding='utf-8') as f:
        return f.read()

system_prompt = _read('system_prompt.txt')
sql_prompt = _read('sql_prompt.txt')
answer_prompt = _read('answer_prompt.txt')
