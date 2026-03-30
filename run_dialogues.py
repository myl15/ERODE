"""
Stage 2: Execute multi-turn dialogues across all (model × scenario × pressure_type) combinations.
Outputs one JSONL file per model in transcripts/.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

from config import MODELS, SCENARIOS_PATH, TRANSCRIPTS_DIR

# --- Provider-specific API wrappers ---

def call_openai(model_cfg, messages):
    """Returns (response_text, usage_dict, model_version)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ[model_cfg.api_key_env])
    response = client.chat.completions.create(
        model=model_cfg.model_id,
        messages=messages,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
    )
    return (
        response.choices[0].message.content,
        {"prompt_tokens": response.usage.prompt_tokens,
         "completion_tokens": response.usage.completion_tokens},
        response.model,  # Exact version string returned by API
    )

def call_anthropic(model_cfg, messages):
    """Returns (response_text, usage_dict, model_version)."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv(model_cfg.api_key_env))
    # Anthropic uses system as a separate param, not in messages
    system_msg = model_cfg.system_prompt
    api_messages = [m for m in messages if m["role"] != "system"]
    response = client.messages.create(
        model=model_cfg.model_id,
        system=system_msg,
        messages=api_messages,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
    )
    return (
        response.content[0].text,
        {"input_tokens": response.usage.input_tokens,
         "output_tokens": response.usage.output_tokens},
        response.model,
    )

def call_google(model_cfg, messages):
    """Returns (response_text, usage_dict, model_version)."""
    import google.generativeai as genai
    genai.configure(api_key=os.getenv(model_cfg.api_key_env))
    model = genai.GenerativeModel(
        model_cfg.model_id,
        system_instruction=model_cfg.system_prompt,
    )
    # Convert messages to Gemini format
    history = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "user" if m["role"] == "user" else "model"
        history.append({"role": role, "parts": [m["content"]]})
    
    # Send last user message, use prior as history
    chat = model.start_chat(history=history[:-1])
    response = chat.send_message(
        history[-1]["parts"][0],
        generation_config=genai.GenerationConfig(
            temperature=model_cfg.temperature,
            max_output_tokens=model_cfg.max_tokens,
        ),
    )
    return (
        response.text,
        {"prompt_tokens": response.usage_metadata.prompt_token_count,
         "completion_tokens": response.usage_metadata.candidates_token_count},
        model_cfg.model_id,
    )

def call_together(model_cfg, messages):
    """Returns (response_text, usage_dict, model_version). Uses Together AI API."""
    from openai import OpenAI  # Together supports OpenAI-compatible API
    client = OpenAI(
        api_key=os.getenv(model_cfg.api_key_env),
        base_url="https://api.together.xyz/v1",
    )
    response = client.chat.completions.create(
        model=model_cfg.model_id,
        messages=messages,
        temperature=model_cfg.temperature,
        max_tokens=model_cfg.max_tokens,
    )
    return (
        response.choices[0].message.content,
        {"prompt_tokens": response.usage.prompt_tokens,
         "completion_tokens": response.usage.completion_tokens},
        response.model,
    )

PROVIDERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
    "together": call_together,
}


def run_dialogue(model_cfg, scenario, pressure_type):
    """
    Execute a single 5-turn dialogue (T0-T5) for one scenario and pressure type.
    Returns a dialogue record dict.
    """
    call_fn = PROVIDERS[model_cfg.provider]
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


def run_all(model_key, scenarios, resume=True):
    """
    Run all scenarios × pressure types for a single model.
    Appends to JSONL file; supports resume by skipping completed (scenario_id, pressure_type) pairs.
    """
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
    
    total = len(scenarios) * len(pressure_types)
    done = len(completed)
    
    with open(outpath, "a") as f:
        for scenario in scenarios:
            for pt in pressure_types:
                key = (scenario["scenario_id"], pt)
                if key in completed:
                    continue
                
                try:
                    result = run_dialogue(model_cfg, scenario, pt)
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    done += 1
                    print(f"[{done}/{total}] {model_key} | {scenario['scenario_id']} | {pt}")
                except Exception as e:
                    print(f"ERROR: {model_key} | {scenario['scenario_id']} | {pt}: {e}")
                    time.sleep(5)  # Back off on errors
                
                time.sleep(0.5)  # Rate limiting buffer


if __name__ == "__main__":
    import sys
    
    model_key = sys.argv[1] if len(sys.argv) > 1 else "gpt4o"
    
    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)
    
    print(f"Running {len(scenarios)} scenarios × 4 pressure types = {len(scenarios)*4} dialogues")
    print(f"Model: {MODELS[model_key].name}")
    run_all(model_key, scenarios)