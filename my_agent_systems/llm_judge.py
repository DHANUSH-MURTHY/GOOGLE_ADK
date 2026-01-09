from google.genai import Client
import os

client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

def judge_response(user_input, model_output, rubric):
    prompt = f"""
You are an evaluator.

User question:
{user_input}

Model response:
{model_output}

Rubric:
{rubric}

Score from 0 to 1 and explain briefly.
"""

    response = client.models.generate_content(
        model=os.getenv("MODEL"),
        contents=prompt
    )

    return response.text
