import os
import json
from django.utils import timezone
from django.conf import settings
from sentence_transformers import SentenceTransformer
from .models import FarmerQuery, Reminder
from .vector_store import (
    load_and_chunk_pdfs,
    get_or_create_faiss,
    chroma_client,
    VectorStore
)
from langchain_community.llms import Ollama as LangchainOllama
from django.contrib.auth.models import User

# ----------------------------------------
# Initialize Vector Store (PDFs + Chroma Cloud)
# ----------------------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v2"
emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

model = "llama3.2"  
ollama_url = "https://dory-renewing-termite.ngrok-free.app/api/chat"

    
hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)



PDF_FOLDER = os.path.join(settings.BASE_DIR, 'pdfs')
chunks = load_and_chunk_pdfs(PDF_FOLDER)
local_store = get_or_create_faiss(chunks, hf_embeddings)
chroma_collection = chroma_client.get_or_create_collection('soybeans')
VS = VectorStore(local_store, chroma_collection)

# ----------------------------------------
# Ollama LLM Client
# ----------------------------------------
import requests

class OllamaClient:
    """
    Wrapper for querying a remote Ollama LLM model via the HTTP API.
    Replace LangchainOllama with direct API calls to the Ollama server.
    """
    def __init__(self, model_name="llama3.2", api_url="https://dory-renewing-termite.ngrok-free.app/api/chat"):
        self.model = model_name
        self.api_url = api_url

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False  # Set to True for streaming
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "No response content.")
        except requests.RequestException as e:
            return f"Error: {e}"

OLLAMA = OllamaClient()

# ----------------------------------------
# Agents Definitions
# ----------------------------------------

class QueryExtractorAgent:
    PROMPT = (
        "You are a Soybean Query Extraction Agent. "
        "Identify the intent and key entities from the farmer's question. "
        "Return pure JSON with 'intent' and 'entities'.\n"
        "Question: '{query}'\n"
        "JSON:"
    )

    def extract(self, raw_query: str) -> dict:
        prompt = self.PROMPT.format(query=raw_query)
        resp = OLLAMA.generate(prompt)

        # DEBUG PRINT
        print("🧠 Ollama response (extractor):", repr(resp))

        try:
            print("successfull extraction")
            return json.loads(resp)
        except json.JSONDecodeError:
            return {
                "intent": "unknown",
                "entities": {},
                "error": "Could not parse Ollama response. Response was empty or invalid."
            }


class InfoRetrievalAgent:
    PROMPT = (
        "You are a Soybean Information Retrieval Agent. Your task is to analyze the following context and extract the most relevant, concise, and practical advice.\n\n"
        "Relevant Entities: {entities}\n\n"
        "Supporting Research Documents:\n{docs}\n\n"
        "Based on the information above, summarize key advice in 2 to 3 well-formed, informative sentences. Focus on clarity and usefulness for the user."
    )

    def __init__(self, vector_store=VS):
        self.vs = vector_store

    def retrieve(self, key_info: dict) -> str:
        query_text = key_info.get('query', '')
        docs = self.vs.search(query_text, top_k=5)
        docs_text = "\n---\n".join(docs)
        prompt = self.PROMPT.format(
            entities=json.dumps(key_info.get('entities', {})),
            docs=docs_text
        )
        return OLLAMA.generate(prompt)

class MemoryAgent:
    def __init__(self, user: User):
        self.user = user

    def history(self):
        return FarmerQuery.objects.filter(user=self.user).order_by('timestamp')


class ProfileAgent:
    PROMPT = (
        "You are a Soybean Profile Agent. "
        "Summarize preferences or recurring themes from the history below.\n"
        "History:\n{history}\n"
        "Profile Summary:"
    )

    def summarize(self, history_qs) -> str:
        items = [f"Q: {q.query_text}\nA: {q.response_text}" for q in history_qs]
        prompt = self.PROMPT.format(history="\n---\n".join(items))
        return OLLAMA.generate(prompt)

class ContextualAdviceAgent:
    PROMPT = (
        "You are the Soybean Farming Assistant Agent. "
        "Your job is to help small-scale farmers by providing clear, accurate, and practical information based on their profile, current question, and expert agronomic advice. "
        "Respond with a valid JSON object that communicates useful insights in a natural, conversational tone. "
        "Return only a valid JSON object with double-quoted keys and values, and ensure all arrays and elements are properly comma-separated and correctly formatted.\n\n"
        "The JSON should contain:\n"
        "- 'answer': a well-written and informative explanation that directly addresses the user's question.\n"
        "- 'recommendations': a list of clear, helpful suggestions or actions the user can take.\n"
        "- 'location': the relevant location if mentioned or inferred from context.\n\n"
        "Profile Summary:\n{profile}\n\n"
        "Current Question:\n{question}\n\n"
        "Advice:\n{advice}\n\n"
        "Only return properly formatted JSON. Do not include any additional commentary outside the JSON structure."
    )

    def __init__(self, extractor, retriever, memory, profile_agent):
        self.extractor = extractor
        self.retriever = retriever
        self.memory = memory
        self.profile = profile_agent

    def advise(self, raw_query: str) -> dict:
        info = self.extractor.extract(raw_query)
        history_qs = self.memory.history()
        profile_summary = self.profile.summarize(history_qs)
        advice_text = self.retriever.retrieve(info)

        print("🧠 Extracted Info:", info)
        print("🧠 Profile Summary:", profile_summary)
        print("🧠 Advice Text:", advice_text)


        prompt = self.PROMPT.format(
            profile=profile_summary,
            question=raw_query,
            advice=advice_text
        )

        response = OLLAMA.generate(prompt)
        print("🧠 Ollama response (advice):",response)

        # if not response:
        #     return {
        #         "answer": "Sorry, I couldn't generate a helpful response at the moment.",
        #         "recommendations": [],
        #         "location": None
        #     }

        try:
            print("successfull advice")
            return response
        except json.JSONDecodeError as e:
            print("❌ JSON parse error:", e)
            print("🔎 Raw Ollama output:", response)
            return {
                "answer": "There was an error processing the response.",
                "recommendations": [],
                "location": None
            }


class ReminderAgent:
    PROMPT = (
        "You are a Soybean Reminder Agent. "
        "Generate a reminder message for upcoming tasks.\n"
        "Location: {location}\n"
        "Season: {season}\n"
        "Reminder:"
    )

    def create_message(self, location: str, season: str) -> str:
        prompt = self.PROMPT.format(location=location, season=season)
        return OLLAMA.generate(prompt)

    def schedule(self, user: User, message: str, send_at: timezone.datetime): 
        Reminder.objects.create(user=user, message=message, send_at=send_at)
