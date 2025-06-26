from langchain_openai import AzureChatOpenAI
import os

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2025-01-01-preview",  # or your api version
    temperature=0.2,
    max_tokens=16000,
    timeout=None, 
    streaming=True
)


# eine sehr einfache basis Funktion, um Antworten zu generieren
def easy_answer(messages):
    ans = llm.invoke(messages).content
    return ans

def stream_answer(messages):
    """
    Generator, der tokenweise Strings liefert.
    Kann direkt an `st.write_stream()` übergeben werden.
    """
    for chunk in llm.stream(messages):
        # `chunk.content` enthält jeweils nur das Delta-Token
        if chunk.content:
            yield chunk.content