# helpers/stream_helper.py

from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def stream_chat_response(user_id: int, messages: list):
    response_iter = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    def token_generator():
        for chunk in response_iter:
            # chunk is a ChatCompletionChunk object
            if chunk.choices and len(chunk.choices) > 0:
                # chunk.choices[0] is a Choice object
                # .delta is a ChoiceDelta object
                # .content is the actual string
                content = chunk.choices[0].delta.content
                if content:
                    yield content

    return token_generator()
