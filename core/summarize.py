import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load environment variables from .env
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


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert content analyst. Summarize the following excerpt clearly and objectively. "
                "Preserve key points, arguments, narrative events, or data without altering the original context.",
            ),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    # Batch processing reduces round-trip overhead compared to sequential loop invocations
    chunk_summaries = map_chain.batch([{"text": chunk} for chunk in chunks])

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert synthesizer. Consolidate the following partial summaries into a clear, "
                "comprehensive final summary using structured bullet points.\n\n"
                "Guidelines:\n"
                "- Naturally adapt your style to the material (e.g., chronological plot/moral for stories, "
                "key takeaways for lectures/tutorials, core discussions/themes for conversations/meetings).\n"
                "- Do not apply corporate, business, or organizational terminology unless the source content is explicitly about business.",
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the transcript, generate a short, descriptive title (maximum 8 words) "
                "that accurately captures the core topic or story. Return only the title text, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    # Slice the first 2000 characters to prevent wasting prompt tokens on title generation
    return title_chain.invoke(transcript[:2000])