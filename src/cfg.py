"""Tiny config + path helpers."""
from __future__ import annotations

import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    with open(path) as fh:
        return yaml.safe_load(fh)


def output_dir(*parts: str) -> str:
    d = os.path.join(REPO_ROOT, "outputs", *parts)
    os.makedirs(d, exist_ok=True)
    return d
