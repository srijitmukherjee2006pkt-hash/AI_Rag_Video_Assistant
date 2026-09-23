import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

# Load environment variables from .env file
load_dotenv()


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set GEMINI_API_KEY in your .env file.")

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=1024,
    )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def get_rag_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an accurate, objective assistant. Answer the user's question 
based ONLY on the transcript context provided below.

If the answer cannot be determined from the context, respond strictly with: 
"I could not find this information in the provided transcript."

Be direct, factual, and concise. When referencing specific speakers, characters, or data, state them clearly.

Context from transcript:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()
    prompt = get_rag_prompt()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)

    llm = get_llm()
    prompt = get_rag_prompt()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer : {answer}")
    return answer