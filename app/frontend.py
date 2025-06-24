import streamlit as st
from answer_chain import easy_answer

# Musst du hier anpassen
#from qa_chain import res
st.image("static/unity-logo.svg")

st.logo("static/unity-logo.svg")
st.title("Änderungsmanagement Agent 🤖")

# Hier lege ich ein Key an in meinem session_state, falls dieser noch nicht angelegt wurde
if "messages" not in st.session_state:
    st.session_state.messages = []

# Alle Nachrichten der Historie werden angezeigt

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# := operator sorgt zusätzlich dafür das der Inhalt nicht None ist!
if prompt := st.chat_input("Ich bin der Tageeschau Bot, stell deine Fragen"):
    # Use Message wird im Container angezeigt
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# eigentlich wäre hier der Langchain aufruf dann geeignet
if prompt!=None:
    response = f"Agent: {easy_answer(prompt)}"
    # Kontext Window des Assistant 
    with st.chat_message("assistant"):
        # Antwort wird angezeigt
        st.markdown(response)
    # Assistant Antwort wird in deinem session_state reingpackt
    st.session_state.messages.append({"role": "assistant", "content": response})



