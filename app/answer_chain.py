from langchain_openai import AzureChatOpenAI
import os

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2025-01-01-preview",  # or your api version
    temperature=0.2,
    max_tokens=16000,
    timeout=None)


# eine sehr einfache basis Funktion, um Antworten zu generieren
def easy_answer(input):
    ans=llm.invoke(input).content
    return ans