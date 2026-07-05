# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTE: This file was rewritten to route every model OpenRouter can host through
# OpenRouter (one central place to manage budget/keys), and to fall back to a
# locally-served vLLM endpoint for models OpenRouter does not host (the Qwen
# math models and your own orchestrator checkpoint). The original NVIDIA-internal
# version is preserved in LLM_CALL.py.nvidia.bak.

import openai
from openai import AzureOpenAI
from openai import OpenAI
import requests
import time
import os
import json
import subprocess
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional

# Root paths come from the environment (see .env). KEYS_DIR caches minted OAuth
# tokens; default it under the repo so nothing is hard-coded to an absolute path.
KEYS_DIR = os.getenv("KEYS_DIR") or os.path.join(
    os.getenv("REPO_PATH", os.path.dirname(os.path.abspath(__file__))), "keys"
)
if not os.path.isdir(KEYS_DIR):
    os.makedirs(KEYS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# OpenRouter routing
# ---------------------------------------------------------------------------
# Every model OpenRouter hosts is routed there, so cost/rate-limits live in one
# dashboard. Anything not in this map (and not matched as a Claude model) falls
# through to a local vLLM server addressed by `model_config` -> that is how the
# Qwen math models and your orchestrator checkpoint run, which also lets you
# test the local serving path.
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# repo model name -> OpenRouter model id. Edit/extend freely; verify slugs at
# https://openrouter.ai/models
OPENROUTER_MODELS = {
    "gpt-5": "openai/gpt-5",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4.1": "openai/gpt-4.1",
    "o3": "openai/o3",
    "o3-mini": "openai/o3-mini",
    "Qwen/Qwen3-32B": "qwen/qwen3-32b",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "qwen/qwen-2.5-coder-32b-instruct",
    "meta-llama/Llama-3.3-70B-Instruct": "meta-llama/llama-3.3-70b-instruct",
    # NOT on OpenRouter -> served locally by vLLM (see evaluation/run_frames.py):
    #   Qwen/Qwen2.5-Math-72B-Instruct
    #   Qwen/Qwen2.5-Math-7B-Instruct
    #   <your orchestrator checkpoint dir>
}


def resolve_openrouter_model(model: str) -> Optional[str]:
    """Return the OpenRouter model id for `model`, or None if it must run locally."""
    if model in OPENROUTER_MODELS:
        return OPENROUTER_MODELS[model]
    ml = model.lower()
    if "claude" in ml:
        if "opus" in ml:
            return os.getenv("OPENROUTER_CLAUDE_OPUS", "anthropic/claude-opus-4")
        return os.getenv("OPENROUTER_CLAUDE_SONNET", "anthropic/claude-sonnet-4")
    return None


def _openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (see .env / .env.example).")
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_APP_URL", "https://github.com/asunyz/ToolOrchestra"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "ToolOrchestra"),
        },
    )


def convert_openai_tools_to_claude(openai_tools: list) -> list:
    claude_tools = []
    for tool in openai_tools:
        if tool.get("type") != "function":
            raise ValueError(f"Unsupported tool type: {tool.get('type')}")

        fn = tool["function"]
        claude_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}})
        })
    return claude_tools


def get_llm_response(model, messages, temperature=1.0, return_raw_response=False,
                     tools=None, show_messages=False, model_type=None, max_length=1024,
                     model_config=None, model_config_idx=0, model_config_path=None,
                     payload=None, openai_client_type='azure_openai', force_local=False,
                     reasoning_effort=None, **kwargs):
    """Unified LLM entry point.

    Routing:
      * If `model` resolves to an OpenRouter model id (and force_local is False),
        the call goes to OpenRouter (OpenAI-compatible Chat Completions).
      * Otherwise the call goes to a local vLLM server described by `model_config`
        (a list of {"ip_addr", "port"} dicts), exactly as before.

    `return_raw_response=True` returns the raw ChatCompletion object (needed for
    tool-call parsing); otherwise the assistant message content string is returned.
    """
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]

    or_model = None if force_local else resolve_openrouter_model(model)

    # -------------------------------------------------------------- OpenRouter
    if or_model is not None:
        if max_length == 1024:            # preserve the original generous default
            max_length = 40000
        extra_body = None
        effort = reasoning_effort or os.getenv("OPENROUTER_REASONING_EFFORT", "high")
        if or_model.startswith("openai/gpt-5") or or_model.startswith("openai/o3"):
            extra_body = {"reasoning": {"effort": effort}}
        client = _openrouter_client()
        answer = ''
        while answer == '':
            try:
                kw = dict(model=or_model, messages=messages,
                          temperature=temperature, max_tokens=max_length)
                if tools:
                    kw["tools"] = tools
                if extra_body:
                    kw["extra_body"] = extra_body
                chat_completion = client.chat.completions.create(**kw)
                if return_raw_response:
                    return chat_completion
                answer = chat_completion.choices[0].message.content
            except Exception as error:
                print('[OpenRouter ERROR]', model, '->', or_model, error)
                time.sleep(30)
        return answer

    # -------------------------------------------------------------- local vLLM
    if not model_config:
        raise ValueError(
            f"Model '{model}' is not on OpenRouter and no local `model_config` was "
            f"provided. Add it to OPENROUTER_MODELS or serve it with vLLM.")
    answer = ''
    while answer == '':
        config_idx = random.choice(range(len(model_config)))
        ip_addr = model_config[config_idx]["ip_addr"]
        port = model_config[config_idx]["port"]
        try:
            vllm_client = OpenAI(api_key="EMPTY", base_url=f"http://{ip_addr}:{port}/v1")
            chat_completion = vllm_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_length,
                temperature=temperature,
                tools=tools,
            )
            if return_raw_response:
                return chat_completion
            answer = chat_completion.choices[0].message.content
        except Exception as error:
            print('[vLLM ERROR]', model, error)
            if os.path.isfile(str(model_config_path)):
                with open(model_config_path) as f:
                    update_model_configs = json.load(f)
                model_config = update_model_configs[model]
            time.sleep(60)
    return answer
