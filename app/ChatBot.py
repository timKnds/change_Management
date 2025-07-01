import streamlit as st
from answer_chain import easy_answer, stream_answer
import io


def extract_csv(text):
    """
    Extrahiert einen CSV-Block aus dem Text (zwischen ```csv ... ```).
    Gibt None zurück, falls kein CSV gefunden.
    """
    import re
    match = re.search(r"```csv\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

st.set_page_config(layout=None)
# Musst du hier anpassen
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
if prompt := st.chat_input("Stelle irgendeine Frage"):
    # Use Message wird im Container angezeigt
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Kontext Window des Assistant
    with st.chat_message("assistant"):
        # Antwort wird angezeigt als stream
        response = st.write_stream(stream_answer(st.session_state.messages))
        csv_data = extract_csv(response)
        if csv_data:
            st.download_button(
                label="CSV herunterladen",
                data=io.BytesIO(csv_data.encode("utf-8")),
                file_name="antwort.csv",
                mime="text/csv")
    # Assistant Antwort wird in deinem session_state reingpackt
    st.session_state.messages.append({"role": "assistant", "content": response})
