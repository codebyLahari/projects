import re
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

def read_file(file_path):
    """Reads the content of the text file."""
    with open(file_path, 'r') as file:
        return file.read()

def chunk_by_paragraph(text, max_chunk_size=500):
    """Chunks the text by paragraph."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]  # Remove empty paragraphs
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

# Example usage
file_path = r"C:\Users\lahar\New folder\Genai\Module-6\india_festivals_paragraphs.txt"
text = read_file(file_path)

# Choose chunking method
chunks_paragraph = chunk_by_paragraph(text)

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose any model from sentence-transformers

# Initialize ChromaDB client
client = chromadb.Client()

# Create or connect to a Chroma collection (database)
collection_name = "india_festivals_collection"
try:
    collection = client.get_or_create_collection(collection_name)
except Exception as e:
    print(f"Error getting/creating collection: {e}")

# Generate embeddings and store in the ChromaDB collection
for i, chunk in enumerate(chunks_paragraph, 1):
    # Generate embeddings for each chunk (paragraph)
    embedding = model.encode(chunk)
    
    # Generate a unique ID for each chunk (could use UUID or chunk number)
    chunk_id = str(uuid.uuid4())  # Using UUID for unique identifiers
    
    # Store the chunk and its embedding in ChromaDB
    try:
        collection.add(
            ids=[chunk_id],  # Unique ID for each document
            documents=[chunk],  # Store the actual text
            metadatas=[{'chunk_id': i}],  # Metadata, like chunk ID or any additional information
            embeddings=[embedding]  # The generated embeddings
        )
    except Exception as e:
        print(f"Error adding chunk {i} to ChromaDB: {e}")

# Optionally, you can search for similar chunks
query = "festival of lights"  # Example query
query_embedding = model.encode(query)

# Search for similar chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5  # Number of results to return
)

# Display the search results
print("\nSearch Results:")
for result in results['documents']:
    print(f"Document: {result}")
