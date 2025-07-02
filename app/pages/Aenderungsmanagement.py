import streamlit as st
from agent.answer_chain import problem_statement,relevant_regularies,relevant_systems,technical_difficulty,cost_calculation

# st.image("./app/static/unity-logo.svg")
# st.logo("./app/static/unity-logo.svg")
st.title("Use Case AI basiertes Änderungsmanagement")



st.set_page_config(layout="wide")  # Ermöglicht mehr Platz


# --- Oberer Bereich in drei Spalten ---
col1, col2, col3 = st.columns([2, 1.5, 1.5])  # linke Spalte breiter

# Linke Spalte – Beschreibung & betroffene Bereiche
with col1:
    # Problemstellung
    st.subheader("Problemstellung")
    problem_input = st.text_area("Beschreibe das Problem", placeholder="z. B. DCDC-Wandler erfüllt nicht die Anforderungen", key="problem_input")

    # Callback-Funktion (wird ausgeführt, wenn auf button geklickt wird)
    def generate_description_callback():
        if st.session_state["problem_input"].strip():
            st.session_state["change_description"] = problem_statement(st.session_state["problem_input"])

    # Änderungsbeschreibung
    st.button("🧠 Änderungsbeschreibung generieren", on_click=generate_description_callback)

    if "change_description" in st.session_state:
        st.text_area("Änderungsbeschreibung", value=st.session_state["change_description"], height=150)

    st.markdown(
    "<hr style='border: 2px solid #edeff3; margin: 30px 0;'>",
    unsafe_allow_html=True)
    ####################################################################################################################################
    # Betroffene Regularien
    st.subheader("Relevante zu beachtende Regularien")

    def generate_search_callback():
         st.session_state["regulatory_answer"]= relevant_regularies(st.session_state["change_description"])

    st.button("🧠 Ermittlung alle relevanter Regularien", on_click=generate_search_callback)
    if "regulatory_answer" in st.session_state:
        st.markdown(st.session_state["regulatory_answer"])


    st.markdown(
    "<hr style='border: 2px solid #edeff3; margin: 30px 0;'>",
    unsafe_allow_html=True)
    #####################################################################################################################################
    # Betroffene Systeme
    st.subheader("⚙️ Betroffene Systeme & Komponenten")

    def generate_system_analysis():
         st.session_state["system_answer"]= relevant_systems(st.session_state["change_description"])

    st.button("🧠 Auswirkungen auf Teilsysteme des Fahrzeuges", on_click=generate_system_analysis)

    if "system_answer" in st.session_state:
        st.markdown(st.session_state["system_answer"])


# Mittlere Spalte – Technische Bewertung
with col2:
    st.subheader("🔧 Technische Bewertung")

    def technical_analysis():
        st.session_state["technical_difficulty"] = technical_difficulty(st.session_state["change_description"])

    st.button("🧠 Technische Analyse", on_click=technical_analysis)
    if "technical_difficulty" in st.session_state:
        st.markdown(st.session_state["technical_difficulty"])

    st.markdown("**Finale technische Bewertung:**")
    st.checkbox("einfach")
    st.checkbox("mittel")
    st.checkbox("aufwändig")
    

# Rechte Spalte – Kostenindikation
with col3:
    st.subheader("💰 Kostenindikation")

    def cost_calculation_click():
        st.session_state["cost_estimation"] = cost_calculation(st.session_state["change_description"])


    st.button("🧠 Kostenabschätzung", on_click=cost_calculation_click)
    if "cost_estimation" in st.session_state:
        st.markdown(st.session_state["cost_estimation"])
    st.text_input("Finale Kosteneingabe (€)", placeholder="z. B. 10.000")
    

# --- Trennung ---
st.markdown(
    "<hr style='border: 2px solid #edeff3; margin: 30px 0;'>",
    unsafe_allow_html=True)

# Unterer Bereich – Änderungsantrag
st.subheader("📄 Änderungsantrag zur Genehmigung")

with st.container():
    col_left, col_right = st.columns(2)

    with col_left:
        st.date_input("Datum")
        st.text_input("Gremium")
        st.text_input("Umsetzungsverantwortlicher")

    with col_right:
        st.selectbox("Empfehlung", ["-", "Empfohlen", "Nicht empfohlen"])
        st.selectbox("Beschluss", ["-", "Angenommen", "Abgelehnt"])

# Einreichen
if st.button("Antrag einreichen"):
    st.success("✅ Änderungsantrag eingereicht.")
