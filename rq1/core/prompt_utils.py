from openai import AzureOpenAI
from core.env_loader import load_env

env = load_env()

client = AzureOpenAI(
    api_version=env['api_version'],
    azure_endpoint=env['endpoint'],
    api_key=env['api_key'],
)

GROUNDED_PROMPT = """
You are an AI assistant that helps users learn from the information found in the source material.
Answer the query concisely using only the sources provided below.
If the answer is longer than 3 sentences, provide a summary.
Answer ONLY with the facts listed in the list of sources below. Cite your source when you answer the question.
If there isn't enough information below, say you don't know.
Do not generate answers that don't use the sources below.
Answer the question directly, without additional explanation, and be as concise as possible. Use maximum 15 words in your response.
Query: {query}
Sources:\n{sources}
"""

def build_prompt_with_sources(query, sources):
    return [
        {
            "role": "user",
            "content": GROUNDED_PROMPT.format(query=query, sources=sources)
        }
    ]


def correctness_labeler(prompt):
    system_prompt = '''Given a question and a ground truth answer, judge the correctness of the candidate response. 
    **Important Definitions**:
    - A response is considered **correct** if it matches the **key information** of the ground truth answer.
    - A response is **incorrect** if it is factually wrong, off-topic, or misleading.

    Return 1 if correct, return 0 if incorrect. Do not return anything else.'''

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_completion_tokens=1,
        temperature=1.0,
        model=env['deployment']
    )

    return response.choices[0].message.content




def incorrect_context_generator(prompt):
    print('incorrect response')
    system_prompt = '''You are an incorrect context generator. Given a question Q, generate a short made up context information that misleads the question from
    giving a correct answer. Make sure your context information does not lead to the correct answer A but rather lead to an incorrect but seemingly correct response.'''
    try: 
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_completion_tokens=800,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=env['deployment']
            )

        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return 'This question is impossible to answer.'