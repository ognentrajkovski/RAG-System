from rag import rag
from langchain_openai import ChatOpenAI

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def query_and_validate(question: str, expected_response: str) -> bool:
    response = rag(question)

    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response,
    )

    model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    evaluation_result = model.invoke(prompt)
    evaluation_result_cleaned = evaluation_result.strip().lower()

    print(prompt)

    if "true" in evaluation_result_cleaned:
        print(f"{evaluation_result_cleaned} is true")
        return True
    elif "false" in evaluation_result_cleaned:
        print(f"{evaluation_result_cleaned} is false")
        return False

    raise ValueError(f"Invalid evaluation result")


def test_glaven_grad():
    assert query_and_validate(
        question="Кој е главен град",
        expected_response="Скопје",
    )
