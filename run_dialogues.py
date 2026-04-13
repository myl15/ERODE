"""
Stage 2: Execute multi-turn dialogues across all (model × scenario × pressure_type) combinations.
Outputs one JSONL file per model in transcripts/.
"""

import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from config import MODELS, SCENARIOS_PATH, TRANSCRIPTS_DIR


def _require_api_key(env_name: str) -> str:
    key = os.getenv(env_name)
    if not key:
        raise ValueError(f"Missing API key: set environment variable {env_name}")
    return key


def _safe_assistant_text(text) -> str:
    return "" if text is None else str(text)


# --- Provider-specific API wrappers ---


def build_google_chat_history(messages):
    """Split OpenAI-style messages for Google GenAI ``generate_content``.

    Returns ``(system_instruction, history, last_user_text)``: ``history`` is a list of
    ``google.genai.types.Content`` for every turn *before* the final user message;
    ``last_user_text`` is that final user's text. The transcript must end with a
    ``user`` message (matches each ``call_google`` invocation in ``run_dialogue``).

    Raises:
        ValueError: if there is no final user message.
    """
    from google.genai import types

    system_instruction = None
    rest = []
    for m in messages:
        if m["role"] == "system":
            system_instruction = m["content"]
            continue
        rest.append(m)
    if not rest or rest[-1]["role"] != "user":
        raise ValueError("Google chat history must end with a user message")
    last = rest[-1]["content"]
    history = []
    for m in rest[:-1]:
        role = "user" if m["role"] == "user" else "model"
        history.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=m["content"])],
            )
        )
    return system_instruction, history, last


def call_openai(model_cfg, messages):
    """Returns (response_text, usage_dict, model_version)."""
    from openai import OpenAI

    client = OpenAI(api_key=_require_api_key(model_cfg.api_key_env))
    response = client.chat.completions.create(
        model=model_cfg.model_id,
        messages=messages,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
    )
    usage = response.usage
    return (
        _safe_assistant_text(response.choices[0].message.content),
        {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        },
        response.model,
    )


def call_anthropic(model_cfg, messages):
    """Returns (response_text, usage_dict, model_version)."""
    import anthropic

    client = anthropic.Anthropic(api_key=_require_api_key(model_cfg.api_key_env))
    system_msg = model_cfg.system_prompt
    api_messages = [m for m in messages if m["role"] != "system"]
    response = client.messages.create(
        model=model_cfg.model_id,
        system=system_msg,
        messages=api_messages,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
    )
    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text
    usage = response.usage
    return (
        _safe_assistant_text(text),
        {
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        },
        response.model,
    )


def call_google(model_cfg, messages):
    """Returns (response_text, usage_dict, model_version).

    Uses client.models.generate_content directly (stateless), which makes
    exactly one API call per invocation — no chat object overhead.
    """
    from google import genai
    from google.genai import types

    _require_api_key(model_cfg.api_key_env)
    client = genai.Client(api_key=os.getenv(model_cfg.api_key_env))

    system_instruction, history, last_user_text = build_google_chat_history(messages)
    contents = list(history) + [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=last_user_text)],
        )
    ]

    # Disable thinking for 2.5 models — default budget is 8192 tokens
    # per call, which adds ~15-20s latency with no benefit for dialogue
    # generation. Only set for 2.5 models; non-thinking models reject it.
    config_kwargs = dict(
        temperature=model_cfg.temperature,
        max_output_tokens=model_cfg.max_tokens,
        system_instruction=system_instruction,
    )
    if "2.5" in model_cfg.model_id:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    response = client.models.generate_content(
        model=model_cfg.model_id,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    meta = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0 if meta else 0
    completion_tokens = getattr(meta, "candidates_token_count", 0) or 0 if meta else 0

    model_version = (
        getattr(response, "model_version", None)
        or getattr(response, "model", None)
        or model_cfg.model_id
    )

    return (
        _safe_assistant_text(getattr(response, "text", None)),
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        model_version,
    )


def call_together(model_cfg, messages):
    """Returns (response_text, usage_dict, model_version). Uses Together AI API."""
    from openai import OpenAI

    client = OpenAI(
        api_key=_require_api_key(model_cfg.api_key_env),
        base_url="https://api.together.xyz/v1",
    )
    response = client.chat.completions.create(
        model=model_cfg.model_id,
        messages=messages,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
    )
    usage = response.usage
    return (
        _safe_assistant_text(response.choices[0].message.content),
        {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        },
        response.model,
    )


_CALL_BY_PROVIDER = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
    "together": call_together,
}


def get_call_fn(model_cfg):
    """Resolve the API call function for a ModelConfig's provider."""
    try:
        return _CALL_BY_PROVIDER[model_cfg.provider]
    except KeyError as e:
        raise ValueError(f"Unknown provider: {model_cfg.provider}") from e


PROVIDERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
    "together": call_together,
}


def run_dialogue(model_cfg, scenario, pressure_type, call_fn=None):
    """
    Execute a single 5-turn dialogue (T0-T5) for one scenario and pressure type.
    Returns a dialogue record dict.
    """
    if call_fn is None:
        call_fn = get_call_fn(model_cfg)
    pressure = scenario["pressure_variants"][pressure_type]

    messages = [{"role": "system", "content": model_cfg.system_prompt}]
    turns = []

    # T0: User poses the question
    t0_content = scenario["question"]
    messages.append({"role": "user", "content": t0_content})
    turns.append({"turn": "T0", "role": "user", "content": t0_content})

    # T1: Model initial response
    t1_text, t1_usage, model_version = call_fn(model_cfg, messages)
    messages.append({"role": "assistant", "content": t1_text})
    turns.append({"turn": "T1", "role": "assistant", "content": t1_text, "usage": t1_usage})

    # T2: First pressure application
    t2_content = pressure["T2"]
    messages.append({"role": "user", "content": t2_content})
    turns.append({"turn": "T2", "role": "user", "content": t2_content})

    # T3: Model response under pressure
    t3_text, t3_usage, _ = call_fn(model_cfg, messages)
    messages.append({"role": "assistant", "content": t3_text})
    turns.append({"turn": "T3", "role": "assistant", "content": t3_text, "usage": t3_usage})
    #time.sleep(4)

    # T4: Escalated pressure
    t4_content = pressure["T4"]
    messages.append({"role": "user", "content": t4_content})
    turns.append({"turn": "T4", "role": "user", "content": t4_content})

    # T5: Final model response
    t5_text, t5_usage, _ = call_fn(model_cfg, messages)
    messages.append({"role": "assistant", "content": t5_text})
    turns.append({"turn": "T5", "role": "assistant", "content": t5_text, "usage": t5_usage})

    return {
        "scenario_id": scenario["scenario_id"],
        "model": model_cfg.name,
        "model_id": model_version,
        "domain": scenario["domain"],
        "pressure_type": pressure_type,
        "ground_truth": scenario.get("ground_truth"),
        "user_claim": scenario["user_claim"],
        "turns": turns,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _run_dialogue_with_retries(model_cfg, scenario, pressure_type, max_retries=5):
    """Run a single dialogue with exponential-backoff retries. Returns result dict or raises."""
    for attempt in range(max_retries):
        try:
            return run_dialogue(model_cfg, scenario, pressure_type)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt * 10  # 10, 20, 40, 80, 160s
            print(f"  RETRY {attempt+1}/{max_retries}: {scenario['scenario_id']} | {pressure_type}: {e}")
            time.sleep(wait)


# Default concurrent workers per provider. Tune based on your rate limits.
# Each worker runs one full dialogue (3 sequential API calls) at a time.
# A dialogue at ~3 calls means N workers ≈ 3N RPM sustained.
_DEFAULT_WORKERS = {
    "openai": 10,     # Paid tiers: 500-10000 RPM
    "anthropic": 8,   # Paid tiers: ~200+ RPM
    "google": 10,     # Paid tiers: 1000+ RPM
    "together": 8,    # Paid tiers: varies
}


def run_all(model_key, scenarios, resume=True, max_workers=None):
    """
    Run all scenarios × pressure types for a single model using concurrent workers.
    Each dialogue runs in its own thread; JSONL writes are serialized via a lock.
    Supports resume by skipping completed (scenario_id, pressure_type) pairs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    model_cfg = MODELS[model_key]
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    outpath = os.path.join(TRANSCRIPTS_DIR, f"{model_key}.jsonl")

    # Load already-completed dialogue IDs for resume
    completed = set()
    if resume and os.path.exists(outpath):
        with open(outpath) as f:
            for line in f:
                rec = json.loads(line)
                completed.add((rec["scenario_id"], rec["pressure_type"]))

    pressure_types = [
        "polite_disagreement",
        "confident_contradiction",
        "appeal_to_authority",
        "emotional_social",
    ]

    # Build task list of (scenario, pressure_type) pairs not yet completed
    tasks = []
    for scenario in scenarios:
        for pt in pressure_types:
            if (scenario["scenario_id"], pt) not in completed:
                tasks.append((scenario, pt))

    total = len(tasks) + len(completed)
    done = len(completed)
    failed = []

    if not tasks:
        print(f"All {total} dialogues already completed for {model_key}.")
        return

    if max_workers is None:
        max_workers = _DEFAULT_WORKERS.get(model_cfg.provider, 5)

    print(f"Dispatching {len(tasks)} dialogues across {max_workers} workers...")

    write_lock = threading.Lock()

    with open(outpath, "a") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_dialogue_with_retries, model_cfg, scenario, pt
                ): (scenario["scenario_id"], pt)
                for scenario, pt in tasks
            }

            for future in as_completed(futures):
                sid, pt = futures[future]
                try:
                    result = future.result()
                    with write_lock:
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        done += 1
                        print(f"[{done}/{total}] {model_key} | {sid} | {pt}")
                except Exception as e:
                    done += 1
                    failed.append((sid, pt, str(e)))
                    print(f"[{done}/{total}] FAILED {model_key} | {sid} | {pt}: {e}")

    if failed:
        print(f"\n{len(failed)} dialogues failed:")
        for sid, pt, err in failed:
            print(f"  {sid} | {pt}: {err}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SycoBench dialogues")
    parser.add_argument("model", nargs="?", default="gpt4o", help="Model key from config.py")
    parser.add_argument("-w", "--workers", type=int, default=None,
                        help="Max concurrent dialogues (default: auto per provider)")
    args = parser.parse_args()

    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)

    # Load environment variables from .env file if it exists
    load_dotenv()

    print(f"Running {len(scenarios)} scenarios × 4 pressure types = {len(scenarios)*4} dialogues")
    print(f"Model: {MODELS[args.model].name}")
    run_all(args.model, scenarios, max_workers=args.workers)