from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
import csv
import time
from dotenv import load_dotenv
# Azure Search connection
load_dotenv()

############ Vektorstore Connection############
azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") 
azure_openai_api_version: str = "2023-05-15"
azure_deployment: str = "embedding"
vector_store_address: str =os.getenv("YOUR_AZURE_SEARCH_ENDPOINT")
vector_store_password: str = os.getenv("YOUR_AZURE_SEARCH_ADMIN_KEY")
# Option 2: Use AzureOpenAIEmbeddings with an Azure account
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=azure_deployment,
    openai_api_version=azure_openai_api_version,
    azure_endpoint=azure_endpoint,
    api_key=azure_openai_api_key,
)
index_name: str = "components-index"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)



# csv lesen
def csv_to_text(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        lines = []
        for row in reader:
            line = ", ".join(f"{key}: {value}" for key, value in row.items())
            lines.append(line)
    return "\n".join(lines)

# connection zum Retriever
retriever = AzureAISearchRetriever(
    content_key="chunk", top_k=5
)

# Retriever für die Komponenten
retriever_components = AzureAISearchRetriever(
    content_key="content", top_k=2, index_name="components-index"
)


# Das eine Funktion, um die retrieveden Dokuemnte als ein gemeinsamen String zu formatieren
def doc_to_string(docs):
    return "\n\n".join(doc.page_content for doc in docs)



llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2025-01-01-preview",  # or your api version
    temperature=0.2,
    max_tokens=16000,
    timeout=None, 
    streaming=True
)

# template definieren für Änderungsbeschreibung
template_change_description= ChatPromptTemplate([
    ("system", "Du bist ein erfahrener Automotive-Experte."
                "Du formulierst Problemstellungen von Ingenieuren in präzise, verständliche Änderungsbeschreibungen um."
                "Die Beschreibung darf maximal 6 Sätze lang sein und soll technisch korrekt und klar formuliert sein."),
    ("human", "{problem}")
])


# Template für RAG Applikation
template_rag_app= PromptTemplate.from_template("""Du bist ein technischer Assistent für Fahrzeugumbauten bei Mercedes-Benz. Deine Aufgabe ist es, Änderungswünsche an Fahrzeugen auf Basis der offiziellen UNECE-Regularien zu prüfen. Deine Informationsquelle ist ausschließlich die Dokumentensammlung „TRANSLATION OF UN REGULATIONS IN THE AREA OF VEHICLE APPROVAL“.

Gehe dabei wie folgt vor:

### Für jede Fahrzeugänderung gib bitte folgende 3 Punkte an:

1. **Relevante Regularien und inhaltliche Auszüge**
   - Identifiziere die zutreffenden UNECE-Regularien für die konkrete Änderung (z. B. „Blinker auf LED umrüsten“ → UNECE Regulation No. 6).
   - Gib zu jeder gefundenen Regulation die relevanten Paragraphen, Artikel oder Anhänge an.
   - Fasse deren Inhalte fachlich korrekt, verständlich und anwendungsbezogen zusammen:
     - z. B. Vorschriften zur Blinkfrequenz, Sichtbarkeitswinkel, Einbauhöhen usw.
     - Verwende Maßeinheiten exakt so, wie sie im Dokument stehen (z. B. „60 bis 120 pro Minute“).
   - Falls eine Regulation keine eindeutige Aussage trifft, erkläre dies sachlich und weise ggf. auf verwandte Vorschriften hin.

2. **Technische Bedeutung für die gewünschte Änderung**
   - Erläutere, welche praktischen Anforderungen sich aus den Regularien für den Umbau ergeben.
   - Benenne ggf. Freigabe-, Prüf- oder Nachweispflichten.
   - Stelle Verbindungen zu sicherheitsrelevanten oder zulassungsrelevanten Aspekten her.

3. **Mögliche Einschränkungen oder alternative Lösungen**
   - Weise auf technische oder rechtliche Grenzen hin, die bei der Umsetzung zu beachten sind.
   - Gib bei Bedarf Vorschläge für zulässige Alternativen auf Basis der Regularien.

### Format

- Gib jeden der drei Punkte nummeriert und mit einer Zwischenüberschrift aus.
- Nutze Bullet-Points für einzelne Vorschrifteninhalte und deren Bedeutung.
- Verwende klare, technische Sprache – wie ein Fachberater bei Mercedes.
- Gib nur Informationen wieder, die direkt aus der UNECE-Dokumentensammlung stammen.

Die vorliegenden Informationen: {context}  
Die zu beantwortende Frage: {question}""")


# Prompt für die betoffenden Systeme
prompt_architecture=PromptTemplate.from_template("""Du bist ein technischer Systemarchitekt bei Mercedes-Benz. Deine Aufgabe ist es, für gegebene Änderungswünsche am Fahrzeug zu analysieren, welche Teilsysteme der Gesamtfahrzeugarchitektur von dieser Änderung betroffen sein könnten.

Die vollständige Fahrzeugarchitektur ist unten angegeben. Sie enthält Hauptsysteme, Subsysteme und Teilsysteme in hierarchischer Form.

---

### Deine Aufgabe:

1. **Identifiziere alle relevanten Teilsysteme**, die potenziell direkt oder indirekt von der beschriebenen Fahrzeugänderung betroffen sind.
2. **Erkläre jeweils kurz, warum dieses Teilsystem betroffen sein könnte** (z. B. durch Funktionsvernetzung, elektrische Anbindung, Regelungseinfluss oder sicherheitskritische Auswirkungen).
3. Gib die Antwort **in strukturierter Listenform** aus:  
   - Teilsystem  
     - Grund der Relevanz

Nutze ausschließlich die Informationen aus der unten angegebenen Fahrzeugarchitektur und logisches technisches Schlussfolgern. Führe keine Annahmen über Funktionen oder Systeme durch, die dort nicht erwähnt sind.

---

### Änderungsbeschreibung:
{change_description}

---

### Fahrzeugarchitektur (vereinfacht, hierarchisch):
{architecture}
""")

prompt_technical_difficulty= PromptTemplate.from_template("""Du bist technischer Experte für Fahrzeugarchitektur bei Mercedes-Benz. Deine Aufgabe ist es, eine geplante Fahrzeugänderung technisch zu bewerten.

Dir stehen folgende Informationen zur Verfügung:
- Eine Beschreibung der geplanten Änderung
- Eine Liste der betroffenen Teilsysteme aus einer vorherigen Analyse
- Technischer Kontext vergangener Änderungen

---

### Deine Aufgabe:

1. **Bewerte die technische Komplexität** der geplanten Änderung basierend auf:
   - Tiefe des Eingriffs in bestehende Systemarchitekturen
   - Anzahl und Kritikalität betroffener Teilsysteme
   - notwendige softwareseitige oder hardwareseitige Anpassungen
   - Systemvernetzung und Abhängigkeiten
   - Auswirkungen auf sicherheitsrelevante Funktionen oder Zulassung

2. Verwende dabei die folgende Klassifikation:

   - **Einfach**: Änderung betrifft isolierte Systeme mit geringem Integrationsaufwand; keine wesentlichen Softwareeingriffe erforderlich.
   - **Mittel**: Änderung betrifft mehrere vernetzte Subsysteme oder erfordert moderate Anpassungen an Steuergeräten, Logik oder Hardware.
   - **Aufwändig**: Änderung erfordert tiefgreifende Eingriffe in sicherheitskritische oder fahrdynamisch relevante Systeme; hoher Aufwand für Integration, Absicherung und Freigabe.

3. Ziehe bei deiner Bewertung Parallelen zum Kontext vergangener Änderungen, falls möglich.

4. Gib deine Antwort in folgendem Format aus:
   - **Technische Bewertung:** einfach / mittel / aufwändig  
   - **Begründung:** Technisch präzise und sachlich formuliert.

---

### Eingabedaten:

**Änderungsbeschreibung:**  
{change_description}

**Betroffene Teilsysteme laut Analyse:**  
{system_analysis}

**Kontext vergangener Änderungen:**  
{context}""")


prompt_cost_estimation = PromptTemplate.from_template("""
Du bist Kostenexperte für Fahrzeugentwicklung bei Mercedes-Benz. Deine Aufgabe ist es, eine geplante Fahrzeugänderung auf Basis technischer Informationen und historischer Erfahrungswerte hinsichtlich der zu erwartenden Umsetzungskosten einzuschätzen.

Dir stehen folgende Informationen zur Verfügung:
- Eine Beschreibung der geplanten Änderung
- Eine Liste betroffener Teilsysteme aus einer vorherigen Analyse
- Technischer Kontext vergangener Änderungen

---

### Deine Aufgabe:

1. **Schätze den voraussichtlichen Kostenaufwand für die Umsetzung der Änderung ab**. Berücksichtige insbesondere:
   - Anzahl und Kritikalität der betroffenen Teilsysteme
   - Umfang von Software- und Hardwareanpassungen
   - Aufwand für Integration, Test, Freigabe und Zertifizierung
   - Komplexität der Systemvernetzung
   - Erfahrungswerte aus vergleichbaren Änderungen im Kontext

2. Gib eine **konkrete Kostenschätzung in Euro oder als sinnvolle Spanne** an.

3. Gib die Antwort im folgenden Format aus:

**Kostenschätzung:** [geschätzter Wert oder Spanne in Euro, z. B. "30.000 – 50.000 €"]  
**Begründung:** [präzise, sachliche Herleitung auf Basis der Eingaben]

---

### Eingabedaten:

**Änderungsbeschreibung:**  
{change_description}

**Betroffene Teilsysteme laut Analyse:**  
{system_analysis}

**Kontext vergangener Änderungen:**  
{context}
""")

# Prompt schreiben, damit er sagen kann, welche Komponenten betroffen sind von der Änderung 
prompt_components = PromptTemplate.from_template("""
Du bist ein technischer Analytiker bei Mercedes-Benz. Du erhältst zwei Eingaben:

1. Eine technische Analyse, die beschreibt, welche Teilsysteme oder Systeme von einer Fahrzeugänderung betroffen sind.
2. Eine vollständige Liste aller Fahrzeugkomponenten im CSV-Format. Jede Zeile enthält:
   - Komponentenname
   - SNS_number
   - Zugeordnetes System

---

### Deine Aufgabe:

- **Lies die technische Analyse sorgfältig.**
- **Finde alle Komponenten in der CSV**, die zu den betroffenen Teilsystemen oder Systemen gehören.
- **Gib ausschließlich diese relevanten Komponenten zurück.**

Die technische Analyse wird separat angezeigt – du brauchst sie nicht erneut auszugeben.

### Ausgabeformat:

- **Komponentenname**
  - SNS_number: …
  - System: …

---

### Eingabedaten:

**Technische Analyse:**  
{system_analysis}

**Fahrzeugkomponenten (CSV):**  
{context}
""")



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


# ich brauche eine Funkion die aus einer Problemstellung eine Änderungsbeschreibung generiert
def problem_statement(input):
    chain=template_change_description | llm | StrOutputParser()
    return chain.invoke({"problem":input})
    

# Ich brauch eine Funktion die alle relevanten Regularien raussucht
def relevant_regularies(input):
    chain = {"context": retriever| doc_to_string, "question": RunnablePassthrough()} | template_rag_app | llm | StrOutputParser()
    return chain.invoke(input)

# Ich brauch eine Funktion die alle betroffenen Systeme wiedergibt und auflistet.
def relevant_systems(input):
    with open("./input_data/architecture.txt") as f:
        architecture=f.read()
    chain= prompt_architecture|llm|StrOutputParser()
    return chain.invoke({"change_description": input, "architecture": architecture})

# def technical_difficulty(input):
#     systems=relevant_systems(input)

#     task= "Bewerte die technische Machbarkeit dieser Änderung basierend auf historischen Änderungen \n\n" + systems
#     chain= {"context": retriever | doc_to_string, "change_description": input, "system_analysis": systems} | prompt_technical_difficulty | llm | StrOutputParser()
#     return chain.invoke(task)
def technical_difficulty(input):
    systems = relevant_systems(input)
    # Kontext für den Retriever erweitern
    context_query = (
        "Bewerte die technische Machbarkeit dieser Änderung basierend auf historischen Änderungen.\n\n"
        + input
    )
    # Hole Kontext-Dokumente mit erweitertem Kontext
    context = doc_to_string(retriever.invoke(context_query))
    chain = {
        "context": lambda _: context,  # wird unten direkt übergeben
        "change_description": lambda _:input,
        "system_analysis": lambda _:systems
    } | prompt_technical_difficulty | llm | StrOutputParser()

    #csv_text= csv_to_text("./input_data/fahrzeugsysteme_komponenten.csv")

    system_analysis = chain.invoke({})

    docs = vector_store.similarity_search(query=system_analysis,k=3,search_type="similarity")

    context=doc_to_string(docs)

    chain_components=prompt_components | llm | StrOutputParser()
    
    
    
    return system_analysis +"\n\n" +chain_components.invoke({"context":context,"system_analysis":system_analysis})

def cost_calculation(input):
    systems = relevant_systems(input)
    # Kontext für den Retriever erweitern
    context_query = (
        "Schätze die Kosten der Änderung basierend auf historischen Änderungen ab.\n\n"
        + input
    )
    # Hole Kontext-Dokumente mit erweitertem Kontext
    context = doc_to_string(retriever.invoke(context_query))
    chain = {
        "context": lambda _: context,  # wird unten direkt übergeben
        "change_description": lambda _:input,
        "system_analysis": lambda _:systems
    } | prompt_cost_estimation | llm | StrOutputParser()
    return chain.invoke({})




# if __name__ == "__main__":
#     docs = vector_store.similarity_search(
#     query="Scheinwerfer finden",
#     k=3,
#     search_type="similarity",)
#     time.sleep(2)
#     print(docs[0].page_content)
#     print(len(docs))
