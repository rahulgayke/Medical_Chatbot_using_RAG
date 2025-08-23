
from src.helper import load_pdf_files, filter_required_info_from_doc, text_splitter, get_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import constants as const
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Extract text from PDF files
extracted_data = load_pdf_files(const.DATA_PATH)

# Filter and split the text into chunks
filtered_docs = filter_required_info_from_doc(extracted_data)

# Split the filtered documents into smaller chunks
text_chunks = text_splitter(filtered_docs, chunk_size=500, chunk_overlap=20)

# Get embeddings for the text chunks
embeddings = get_embeddings()

# Initialize Pinecone client
pinecone_api_key = PINECONE_API_KEY
pc_client = Pinecone(api_key=pinecone_api_key)

# Create the Pinecone index
index_name = "medical-chatbot-knowledge-base"
if not pc_client.has_index(index_name):
    pc_client.create_index(
        name=index_name,
        dimension=len(embeddings.embed_query("test")),
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc_client.Index(index_name)

# Store all the text chunks in the Pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
