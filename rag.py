import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question: {question}
based on the following context:

{context}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()
    query_text = args.query_text
    rag(query_text)

def rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=3)
    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query_text)

    model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    response = model.invoke([HumanMessage(content=prompt)])

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response.content}\nSources: {sources}"
    print(formatted_response)
    return response.content

if __name__ == "__main__":
    main()
