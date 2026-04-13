from run_dialogues import build_google_chat_history


def test_build_google_chat_history_basic():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello?"},
        {"role": "assistant", "content": "Hi there."},
        {"role": "user", "content": "Follow-up."},
    ]
    system_instruction, history, last = build_google_chat_history(messages)
    assert system_instruction == "You are helpful."
    assert last == "Follow-up."
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].parts[0].text == "Hello?"
    assert history[1].role == "model"
    assert history[1].parts[0].text == "Hi there."


def test_build_google_chat_history_requires_final_user():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    try:
        build_google_chat_history(messages)
    except ValueError as e:
        assert "user" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")
