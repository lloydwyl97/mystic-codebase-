import openai
import os


def generate_strategy_code(prompt):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a crypto trading strategy coder.",
            },
            {
                "role": "user",
                "content": f"Generate a Python strategy: {prompt}",
            },
        ],
    )
    return response.choices[0].message.content
