import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import gradio as gr
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from serpapi.google_search import GoogleSearch
import re

# Authentification avec le token Hugging Face
login("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")

# Charger les modèles
dpr_model = SentenceTransformer("all-mpnet-base-v2")
generator_model_name = "t5-large"
generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)

# Corpus des documents (local)
corpus_of_documents = [
    "Le Lac Rose, également connu sous le nom de Lac Retba, est célèbre pour sa couleur rose unique due à une forte concentration de sel et de bactéries. C’est un lieu prisé par les touristes et les photographes pour son paysage exceptionnel.",
    "L’île de Gorée est un site historique et un symbole de la traite des esclaves. Classée au patrimoine mondial de l’UNESCO, elle abrite la Maison des Esclaves et de nombreux bâtiments coloniaux colorés.",
    "Le désert de Lompoul est une petite étendue de dunes de sable doré située entre Dakar et Saint-Louis. Il est apprécié pour ses paysages désertiques et ses campements sous des tentes traditionnelles.",
    "La ville de Saint-Louis est une ancienne capitale coloniale au charme nostalgique. Son architecture coloniale, son pont Faidherbe et son ambiance culturelle en font une destination prisée.",
    "Le parc national de Niokolo-Koba est une réserve naturelle au sud-est du Sénégal, reconnue pour sa biodiversité, avec des espèces comme les lions, les éléphants, les antilopes et les chimpanzés.",
    "Le village des tortues de Sangalkam est un refuge pour plusieurs espèces de tortues menacées. Les visiteurs peuvent y observer de près la plus grande tortue d’Afrique, la tortue géante sillonnée.",
    "La Maison Senghor, située à Dakar, est un musée dédié au premier président du Sénégal, Léopold Sédar Senghor. Elle présente des objets personnels et retrace l’histoire de ce grand poète et homme d’État.",
    "Le village des Arts de Dakar est un centre culturel où de nombreux artistes travaillent et exposent leurs œuvres. C’est un lieu dynamique qui permet aux visiteurs de découvrir la créativité sénégalaise.",
    "Le Phare des Mamelles est un phare emblématique situé sur la colline des Mamelles à Dakar. Il offre une vue panoramique sur la ville et la côte atlantique, et est un endroit populaire pour admirer le coucher de soleil.",
    "La réserve de Bandia, située près de Dakar, est une réserve faunique abritant des espèces sauvages telles que les girafes, les rhinocéros et les buffles. C'est une destination prisée pour les safaris en véhicule tout-terrain.",
    "La mosquée de Touba est l'un des plus grands lieux de culte d'Afrique de l'Ouest. Elle est le centre spirituel de la confrérie mouride, fondée par Cheikh Ahmadou Bamba, et attire des milliers de pèlerins chaque année.",
    "Le monastère de Keur Moussa, situé près de Dakar, est une abbaye bénédictine réputée pour ses messes chantées, où les chants grégoriens sont accompagnés par des instruments traditionnels africains comme la kora.",
    "Le vieux Rufisque est un quartier historique qui reflète l'histoire commerciale et coloniale du Sénégal. Les bâtiments anciens et les vestiges de l’époque coloniale font de ce quartier un lieu riche en histoire.",
    "La ville de Ziguinchor, capitale de la région de Casamance, est célèbre pour son ambiance paisible, ses mangroves, et son architecture coloniale. Elle est un point de départ pour découvrir la culture diola.",
    "Le site mégalithique de Sine Ngayène est classé au patrimoine mondial de l'UNESCO. C'est un site archéologique fascinant avec des cercles de pierres érigés il y a plus de mille ans, un mystère encore non résolu."
]

# Embedding et indexation avec FAISS
document_embeddings = dpr_model.encode(corpus_of_documents)
dimension = document_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(document_embeddings))

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Fonction de recherche web avec SerpAPI
def web_search(query, api_key, top_k=3):
    search = GoogleSearch({"q": query, "api_key": api_key})
    results = search.get_dict()
    return [clean_text(result['snippet']) for result in results.get('organic_results', [])[:top_k]]

# Fonction de récupération des documents locaux
def retrieve_documents(query, model, faiss_index, corpus, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [corpus[idx] for idx in indices[0]]

# Génération de réponse avec contexte
def generate_answer(query, retrieved_docs, generator_model, generator_tokenizer):
    context = " ".join(retrieved_docs)
    input_text = f"Question: {query}\nContext: {context}"
    inputs = generator_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator_model.generate(**inputs, max_length=150)
    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Fonction principale du chatbot
def chatbot(query):
    # Récupérer des documents locaux et en ligne
    retrieved_local = retrieve_documents(query, dpr_model, faiss_index, corpus_of_documents)
    web_results = web_search(query, api_key="a4304a16f16cc45f1488979e5c854e12ec7d4c28809f99f08ec77d7d797ac89f")
    
    retrieved_docs = retrieved_local + web_results
    answer = generate_answer(query, retrieved_docs, generator_model, generator_tokenizer)
    
    return answer, retrieved_docs

# Interface Gradio avec design personnalisé
with gr.Blocks(css="""
    .block {margin: 20px; padding: 20px; background-color: #f9f9f9; border-radius: 8px;}
    .title {font-size: 24px; color: #003366; text-align: center; margin-bottom: 10px;}
    .image-box img {width: 100%; border-radius: 8px; margin-top: 10px;}
""") as interface:
    gr.Markdown(
        """
        <div class="title">
            🗺️ Chatbot Touristique Sénégal
        </div>
        <p style="text-align: center;">Découvrez les merveilles du Sénégal : ses lieux emblématiques, activités incontournables et bien plus encore.</p>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📍 Carte des lieux touristiques")
            gr.Image("téléchargement.png", label="Carte des lieux", interactive=False)
        with gr.Column(scale=2):
            user_input = gr.Textbox(lines=3, label="Posez votre question", placeholder="Ex : Que visiter à Dakar ?")
            submit_button = gr.Button("Demander")

    with gr.Row():
        output_answer = gr.Textbox(label="Réponse", lines=3)
    
    with gr.Row():
        gr.Markdown("### 📜 Sources Utilisées")
        output_sources = gr.Textbox(label="Sources utilisées", interactive=False)
    
    with gr.Row():
        gr.Markdown("### 🌟 Suggestion")
        gr.Markdown("Pour une expérience optimale, demandez des informations sur des lieux spécifiques comme le Lac Rose, Gorée ou Lompoul.")

    submit_button.click(chatbot, inputs=[user_input], outputs=[output_answer, output_sources])

interface.launch()
