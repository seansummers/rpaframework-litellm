from RPA.LiteLLM import LiteLLM
from unittest.mock import patch


@patch("litellm.text_completion")
def test_completion_create(mock):
    mock.return_value = {"choices": [{"text": "Hello World"}]}
    lib = LiteLLM()
    response = lib.completion_create("text-davinci-003", "Foobar")
    assert response == "Hello World"


@patch("litellm.completion")
def test_chat_completion_create(mock):
    mock.return_value = {
        "choices": [{"message": {"content": "Hello World", "role": "assistant"}}]
    }
    lib = LiteLLM()
    response = lib.chat_completion_create("text-davinci-003", user_content="Foobar")
    assert response == [
        "Hello World",
        [
            {"content": "Foobar", "role": "user"},
            {"content": "Hello World", "role": "assistant"},
        ],
    ]
