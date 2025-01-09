import os
import time
import pymongo
from pymongo.errors import OperationFailure
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# MongoDB connection details
MONGO_HOST = "localhost"  # Assuming MongoDB is running locally
MONGO_PORT = 27017
MONGO_DB = "mydatabase"  # Replace with your database name
MONGO_COLLECTION = "india_festivals_collection"

# Connection string
connection_string = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/"

# Connect to MongoDB
try:
    client = pymongo.MongoClient(connection_string)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]
    print("Connected to MongoDB successfully!")
except OperationFailure as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# Initialize SentenceTransformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Groq API Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Corrected endpoint
GROQ_API_KEY = "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"  # Replace with your actual API key
GROQ_MODEL = "llama-3.3-70b-versatile"  # Replace with the specific model name

# Path to the text file to monitor
TEXT_FILE_PATH = r"C:\Users\Public\RAG\Module-6(rag)\india_festivals_paragraphs.txt"  # Replace with your text file path

# Function to read content from the file
def read_file():
    with open(TEXT_FILE_PATH, 'r') as file:
        return file.read()

# Function to chunk the text and embed it
def process_text_and_embed(text):
    # Chunk the text (this can be refined further, for now we split by lines)
    chunks = text.split('\n')  # Example: splitting by line breaks (can adjust as needed)
    embeddings = embedding_model.encode(chunks)
    return chunks, embeddings

# Function to store new chunks in MongoDB
def store_in_mongodb(chunks, embeddings):
    # Insert new chunks and their embeddings into the database
    documents = [{"text": chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]
    collection.insert_many(documents)
    print(f"Inserted {len(documents)} new chunks into the database.")

# Function to fetch relevant content based on the user query
def fetch_relevant_content(user_query, top_k=3):
    query_embedding = embedding_model.encode(user_query).reshape(1, -1)
    documents = list(collection.find({}, {"_id": 0, "text": 1, "embedding": 1}))
    if not documents:
        print("No documents found in the database.")
        return []

    texts = [doc["text"] for doc in documents]
    embeddings = [doc["embedding"] for doc in documents]
    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant_chunks = [{"text": texts[i], "score": similarities[i]} for i in top_indices]
    return relevant_chunks

# Function to generate answer using Groq API
def generate_answer(query, relevant_chunks):
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant.give me the answer in detailed explanation with two paragraphs."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
        ],
        "max_tokens": 150
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                print("Unexpected response format:", result)
                return "Failed to generate an answer due to unexpected response."
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return "Failed to generate an answer due to API error."
    except requests.RequestException as e:
        print(f"Error during API request: {e}")
        return "Failed to generate an answer due to request error."

# Function to monitor the text file and update the database
def monitor_and_update_file():
    last_modified_time = os.path.getmtime(TEXT_FILE_PATH)

    while True:
        # Check if the file has been modified
        current_modified_time = os.path.getmtime(TEXT_FILE_PATH)

        if current_modified_time > last_modified_time:
            print("New content detected, processing...")
            new_content = read_file()

            chunks, embeddings = process_text_and_embed(new_content)
            store_in_mongodb(chunks, embeddings)

            last_modified_time = current_modified_time

        # Wait for 60 seconds before checking again
        time.sleep(60)

# Example usage for answering queries
def handle_query():
    user_query = "why do we celebrate diwali?"
    relevant_chunks = fetch_relevant_content(user_query)

    if relevant_chunks:
        print("\nRelevant Chunks:")
        for chunk in relevant_chunks:
            print(f"Text: {chunk['text']}\nScore: {chunk['score']}\n")
        answer = generate_answer(user_query, relevant_chunks)
        print("\nGenerated Answer:")
        print(answer)
    else:
        print("No relevant content found.")

if __name__ == "__main__":
    # Start monitoring the file for new content in the background
    import threading
    monitoring_thread = threading.Thread(target=monitor_and_update_file)
    monitoring_thread.daemon = True
    monitoring_thread.start()

    # Handle user queries
    handle_query()
