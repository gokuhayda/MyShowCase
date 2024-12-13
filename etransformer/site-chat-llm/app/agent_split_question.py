# Importar bibliotecas necessárias
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Método 1: Básico (string split)
def split_basic(question):
    return question.split("?")[:-1]

# Método 2: Gramatical (Spacy)
def split_grammatical(question):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)
    subquestions = []
    current_subq = []

    for token in doc:
        current_subq.append(token.text)
        if token.text in ["and", "or", ",", "?"]:
            subquestions.append(" ".join(current_subq).strip())
            current_subq = []
    if current_subq:
        subquestions.append(" ".join(current_subq).strip())
    return [subq.strip() for subq in subquestions if subq]

# Método 3: Transformers (T5)
def split_transformers(question):
    summarizer = pipeline("summarization", model="t5-small")
    prompt = f"Break the following question into subquestions: {question}"
    result = summarizer(prompt, max_length=50, min_length=10, do_sample=False)
    return result[0]["summary_text"].split(". ")

# Método 4: Embeddings (SentenceTransformers)
def split_embeddings(question):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = question.split("?")
    embeddings = model.encode(chunks)
    similarity_matrix = util.cos_sim(embeddings, embeddings)
    clusters = []
    threshold = 0.7
    for i, sim in enumerate(similarity_matrix):
        cluster = [chunks[j] for j in range(len(sim)) if sim[j] > threshold]
        clusters.append(cluster)
    subquestions = [" ".join(cluster) for cluster in clusters if cluster]
    return subquestions
    
def split_into_subquestions(question):
    # Regras baseadas em heurísticas
    if len(question.split()) <= 10:
        # Perguntas curtas: método básico
        return split_basic(question)
    elif "and" in question or "or" in question or "," in question:
        # Perguntas com conjunções: método gramatical
        return split_grammatical(question)
    elif len(question.split()) > 30:
        # Perguntas muito longas: método com Transformers
        return split_transformers(question)
    else:
        # Padrão geral: embeddings
        return split_embeddings(question)
