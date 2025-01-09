import pymongo
from pymongo.errors import OperationFailure, ConfigurationError
from sentence_transformers import SentenceTransformer
import uuid

# MongoDB connection details
MONGO_HOST = "localhost"  # Assuming MongoDB is running locally
MONGO_PORT = 27017
MONGO_DB = "mydatabase"  # Replace with your database name
MONGO_COLLECTION = "india_festivals_collection"

# Connection string (no username or password)
connection_string = f"mongodb://{MONGO_HOST}:{MONGO_PORT}/"

# Connect to MongoDB
try:
    client = pymongo.MongoClient(connection_string)
    # Check if connection is successful
    client.admin.command('ping')
    print("Connected to MongoDB successfully!")
except (OperationFailure, ConfigurationError) as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# Ensure the database and collection exist
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]
print(f"Database '{MONGO_DB}' and collection '{MONGO_COLLECTION}' are ready.")

# Example data processing
def read_file(file_path):
    """Reads the content of the text file."""
    with open(file_path, 'r') as file:
        return file.read()

def chunk_by_paragraph(text, max_chunk_size=500):
    """Chunks the text by paragraph."""
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

# File path
file_path = r"C:\Users\Public\RAG\Module-6(rag)\india_festivals_paragraphs.txt"
text = read_file(file_path)

# Chunk the text
chunks = chunk_by_paragraph(text)

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings and store in MongoDB
for i, chunk in enumerate(chunks, 1):
    embedding = model.encode(chunk).tolist()  # Convert embedding to a list for MongoDB compatibility
    document = {
        "chunk_id": i,
        "text": chunk,
        "embedding": embedding
    }
    try:
        collection.insert_one(document)
        print(f"Stored chunk {i} in MongoDB.")
    except Exception as e:
        print(f"Error storing chunk {i}: {e}")

# Example query
query = "festival of lights"
query_embedding = model.encode(query).tolist()

print("\nData stored successfully!")

