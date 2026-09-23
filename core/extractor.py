import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load environment variables from .env file
load_dotenv()


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set GEMINI_API_KEY in your .env file.")

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0.1,
        max_output_tokens=1024,
    )


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "Analyze the provided transcript and extract all explicit action items, practical takeaways, "
        "or instructions.\n"
        "- If the text is a meeting or conversation: identify assigned tasks, owners, and deadlines.\n"
        "- If the text is an educational video, tutorial, or narrative/story: list concrete recommendations, "
        "practices, moral lessons, or step-by-step guidance.\n\n"
        "Format as a numbered list. If no applicable takeaways or action items exist, return 'No actionable takeaways found.'"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "Analyze the provided transcript and extract key decisions, central conclusions, "
        "or major thematic turning points.\n"
        "- For meetings/discussions: note agreements or resolved paths forward.\n"
        "- For lectures/talks: note primary claims, proved concepts, or conclusions.\n"
        "- For narratives/stories: note pivotal character choices or the ultimate resolution.\n\n"
        "Format as a numbered list. If none are present, return 'No key conclusions or decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "Analyze the transcript and list all open, unresolved questions, debates, "
        "or suggested areas for future exploration mentioned by the speaker(s).\n\n"
        "Format as a numbered list. If no unresolved questions or follow-ups are raised, "
        "return 'No open questions found.'"
    )
    return chain.invoke(transcript)