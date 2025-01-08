import time
import uuid
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb


# Function to read the file content
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""


# Function to calculate the hash of file content
def calculate_hash(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# Function to chunk text by paragraph
def chunk_by_paragraph(text, max_chunk_size=500):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chunk_size:
            current_chunk += "\n\n" + paragraph
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Load the lightweight model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")


# Function to get embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.numpy()


# ChromaDB Initialization
client = chromadb.Client()
collection_name = "dynamic_festival_collection"
collection = client.get_or_create_collection(collection_name)


# Function to process and store content if new content is detected
def process_and_store(file_path, previous_hash):
    print("\nChecking for new content...")
    new_content = read_file(file_path)
    if not new_content:
        print("File is empty or not found. Skipping processing.")
        return previous_hash

    # Calculate the current hash of the file content
    current_hash = calculate_hash(new_content)
    if current_hash == previous_hash:
        print("No changes detected. Skipping processing.")
        return previous_hash

    print("Changes detected. Processing new content...\n")
    print(f"File Hash: {current_hash}")

    # Step 1: Chunk the new content
    chunks = chunk_by_paragraph(new_content)
    print(f"Total Chunks Generated: {len(chunks)}")

    # Step 2: Generate embeddings and store in ChromaDB
    embeddings = get_embeddings(chunks)  # Optimized embeddings
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
        chunk_id = str(uuid.uuid4())
        try:
            collection.add(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[{'chunk_number': i}],
                embeddings=[embedding]
            )
            print(f"Chunk {i} stored successfully: {chunk[:50]}...")
        except Exception as e:
            print(f"Error storing chunk {i}: {e}")

    print("New content processing and storing completed.\n")
    return current_hash


# File Monitor Class
class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, file_path):
        self.file_path = file_path
        self.previous_hash = calculate_hash(read_file(file_path))

    def on_modified(self, event):
        if event.src_path == self.file_path:
            self.previous_hash = process_and_store(self.file_path, self.previous_hash)


# Main Function to Start Monitoring
def monitor_file(file_path):
    event_handler = FileChangeHandler(file_path)
    observer = Observer()
    observer.schedule(event_handler, path=file_path, recursive=False)
    observer.start()
    print(f"Monitoring changes in: {file_path}")

    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# Specify the file to monitor
file_path = r"C:\Users\lahar\New folder\Genai\Module-6\india_festivals_paragraphs.txt"
monitor_file(file_path)
