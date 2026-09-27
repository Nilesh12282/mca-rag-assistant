import os
import fitz  # PyMuPDF
import hashlib
import requests
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDER = os.path.abspath("data/mca_documents")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hr-policies")

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

OLLAMA_URL = "http://localhost:11434"

# Chunk configuration
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400


# ============================================================
# VALIDATE API KEY
# ============================================================

if not PINECONE_API_KEY:
    raise ValueError(
        "❌ PINECONE_API_KEY is not set in the .env file."
    )


# ============================================================
# SETUP PINECONE
# ============================================================

print("\n🔌 Connecting to Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [i.name for i in pc.list_indexes()]


# ============================================================
# CREATE / VALIDATE INDEX
# ============================================================

if INDEX_NAME in existing_indexes:

    print(f"✅ Pinecone index found: {INDEX_NAME}")

    info = pc.describe_index(INDEX_NAME)

    if info.dimension != EMBEDDING_DIM:

        print(
            f"⚠️ Index '{INDEX_NAME}' has dimension "
            f"{info.dimension}, expected {EMBEDDING_DIM}."
        )

        print(f"🗑️ Deleting incompatible index '{INDEX_NAME}'...")

        pc.delete_index(INDEX_NAME)

        print(f"🆕 Creating index '{INDEX_NAME}'...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

        print(
            f"✅ Created index '{INDEX_NAME}' "
            f"with dimension {EMBEDDING_DIM}"
        )

else:

    print(
        f"🆕 Creating new Pinecone index '{INDEX_NAME}'..."
    )

    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

    print(
        f"✅ Created index '{INDEX_NAME}' successfully"
    )


# Connect to Pinecone index
index = pc.Index(INDEX_NAME)


# ============================================================
# DELETE OLD EMBEDDINGS
# ============================================================

def delete_old_embeddings():
    """
    Delete all existing vectors from all Pinecone namespaces
    before uploading fresh MCA document embeddings.
    """

    try:

        print(
            "\n🗑️ Checking existing embeddings in Pinecone..."
        )

        stats = index.describe_index_stats()

        namespaces = getattr(stats, "namespaces", None)

        if not namespaces:

            print(
                "ℹ️ No existing embeddings found. "
                "Starting with an empty index."
            )

            return

        deleted_any = False

        for namespace, namespace_stats in namespaces.items():

            vector_count = getattr(
                namespace_stats,
                "vector_count",
                0
            )

            if vector_count > 0:

                print(
                    f"🗑️ Deleting {vector_count} vectors "
                    f"from namespace '{namespace}'..."
                )

                index.delete(
                    delete_all=True,
                    namespace=namespace
                )

                deleted_any = True

        if deleted_any:

            print(
                "✅ Old embeddings deleted successfully."
            )

        else:

            print(
                "ℹ️ No existing embeddings found."
            )

    except Exception as e:

        print(
            f"❌ Error deleting old embeddings: {e}"
        )

        raise


# ============================================================
# GENERATE EMBEDDING USING OLLAMA
# ============================================================

def get_embedding(text: str):
    """
    Generate embedding using local Ollama
    and nomic-embed-text model.
    """

    try:

        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",

            json={
                "model": EMBEDDING_MODEL,
                "prompt": text
            },

            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        embedding = data.get("embedding")

        if not embedding:

            print(
                "❌ Ollama returned an empty embedding."
            )

            return None

        return embedding

    except Exception as e:

        print(
            f"❌ Error generating embedding: {e}"
        )

        return None


# ============================================================
# CREATE OVERLAPPING CHUNKS
# ============================================================

def create_chunks(text: str):
    """
    Split document text into overlapping chunks.

    Example:

    Chunk 1 -> characters 0 to 2000
    Chunk 2 -> characters 1600 to 3600
    Chunk 3 -> characters 3200 to 5200

    This overlap helps preserve information when
    subjects/units fall near chunk boundaries.
    """

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + CHUNK_SIZE

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        # Move forward while keeping overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ============================================================
# PROCESS AND UPLOAD DOCUMENTS
# ============================================================

def load_and_embed_docs(folder_path: str):
    """
    Load all MCA PDFs from the folder,
    extract text,
    create overlapping chunks,
    generate embeddings,
    and upload them to Pinecone.
    """

    if not os.path.exists(folder_path):

        print(
            f"❌ Folder not found: {folder_path}"
        )

        return

    print(
        f"\n📂 Loading MCA documents from: "
        f"{folder_path}"
    )

    pdf_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:

        print(
            "⚠️ No PDF files found."
        )

        return

    print(
        f"📚 Found {len(pdf_files)} PDF file(s)."
    )


    # ========================================================
    # PROCESS EACH PDF
    # ========================================================

    for file in pdf_files:

        pdf_path = os.path.join(
            folder_path,
            file
        )

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"📘 Processing: {file}"
        )

        print(
            f"{'=' * 60}"
        )

        try:

            # ------------------------------------------------
            # OPEN PDF
            # ------------------------------------------------

            doc = fitz.open(pdf_path)

            text_parts = []

            for page_number, page in enumerate(
                doc,
                start=1
            ):

                page_text = page.get_text("text")

                if page_text.strip():

                    text_parts.append(
                        page_text
                    )

            doc.close()

            # Combine all pages
            text = "\n\n".join(text_parts)

            if not text.strip():

                print(
                    f"⚠️ No text extracted from {file}"
                )

                continue

            print(
                f"📄 Extracted characters: "
                f"{len(text)}"
            )


            # ------------------------------------------------
            # CREATE OVERLAPPING CHUNKS
            # ------------------------------------------------

            chunks = create_chunks(text)

            print(
                f"🧩 Created {len(chunks)} chunks"
            )

            print(
                f"📏 Chunk size: {CHUNK_SIZE}"
            )

            print(
                f"🔄 Chunk overlap: {CHUNK_OVERLAP}"
            )


            # ------------------------------------------------
            # GENERATE EMBEDDINGS
            # ------------------------------------------------

            uploaded_count = 0

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                if not chunk.strip():

                    continue

                print(
                    f"   🔹 Embedding chunk "
                    f"{chunk_number}/{len(chunks)}..."
                )

                embedding = get_embedding(chunk)

                if embedding is None:

                    print(
                        f"   ⚠️ Skipping chunk "
                        f"{chunk_number} "
                        f"because embedding failed."
                    )

                    continue


                # ------------------------------------------------
                # UNIQUE VECTOR ID
                # ------------------------------------------------

                vector_id = hashlib.md5(
                    f"{file}:{chunk_number}:{chunk}".encode(
                        "utf-8"
                    )
                ).hexdigest()


                # ------------------------------------------------
                # UPLOAD TO PINECONE
                # ------------------------------------------------

                index.upsert(
                    vectors=[
                        {
                            "id": vector_id,

                            "values": embedding,

                            "metadata": {
                                "text": chunk,
                                "source": file,
                                "chunk": chunk_number
                            }
                        }
                    ]
                )

                uploaded_count += 1


            # ------------------------------------------------
            # FILE COMPLETE
            # ------------------------------------------------

            print(
                f"\n✅ Uploaded "
                f"{uploaded_count}/{len(chunks)} chunks "
                f"from: {file}"
            )

        except Exception as e:

            print(
                f"❌ Error processing {file}: {e}"
            )


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print(
        "\n🚀 Starting MCA Assistant "
        "document embedding process..."
    )

    print(
        f"📁 Data folder: {DATA_FOLDER}"
    )

    print(
        f"🧠 Embedding model: {EMBEDDING_MODEL}"
    )

    print(
        f"📐 Embedding dimension: {EMBEDDING_DIM}"
    )

    print(
        f"📦 Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"🔄 Chunk overlap: {CHUNK_OVERLAP}"
    )


    # ========================================================
    # STEP 1: DELETE OLD VECTORS
    # ========================================================

    delete_old_embeddings()


    # ========================================================
    # STEP 2: EMBED + UPLOAD CURRENT PDFs
    # ========================================================
w
    load_and_embed_docs(DATA_FOLDER)


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n🎯 MCA document embedding and "
        "Pinecone upload completed successfully!"
    )