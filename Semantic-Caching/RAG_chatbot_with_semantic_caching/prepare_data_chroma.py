"""
Data Preparation Script for TaskFlow RAG Chatbot with ChromaDB

This script:
1. Loads TaskFlow FAQ data as simulated documents
2. Creates a persistent ChromaDB vectorstore
3. Prepares initial semantic cache from seed data
"""

import os
import csv
import argparse
import traceback
from langchain_core.documents import Document
from document_store_chroma import DocumentVectorStore
from semantic_cache import SemanticCache
from pyprojroot import here
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def load_faq_as_documents(faq_file: str) -> list[Document]:
    """
    Load FAQ data and convert to Document objects.

    Each FAQ entry becomes a document with the question and answer combined.
    This simulates a knowledge base for the RAG system.
    """
    print("\n📚 Loading FAQ data as documents...")
    print("-" * 70)

    documents = []

    with open(faq_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            question = row['question']
            answer = row['answer']

            # Create a document combining question and answer
            # This simulates how documentation typically looks
            content = f"Q: {question}\n\nA: {answer}"

            doc = Document(
                page_content=content,
                metadata={
                    "source": "taskflow_faq",
                    "question": question,
                    "doc_id": f"faq_{idx}",
                    "type": "faq"
                }
            )
            documents.append(doc)

    print(f"  ✓ Loaded {len(documents)} FAQ entries as documents")
    return documents


def load_documentation(doc_file: str) -> list[Document]:
    """
    Load the main TaskFlow documentation file.
    """
    print("\n📖 Loading main documentation...")
    print("-" * 70)

    with open(doc_file, 'r', encoding='utf-8') as f:
        content = f.read()

    doc = Document(
        page_content=content,
        metadata={
            "source": "taskflow_docs",
            "doc_id": "main_docs",
            "type": "documentation"
        }
    )

    print(f"  ✓ Loaded main documentation ({len(content)} characters)")
    return [doc]


def create_document_vectorstore(
    persist_dir: str = str(here("data/chroma_db")),
    collection_name: str = "taskflow_docs",
    force_recreate: bool = False
):
    """
    Create or load the document vectorstore with ChromaDB.

    Args:
        persist_dir: Directory to persist ChromaDB
        collection_name: Name of the collection
        force_recreate: If True, delete existing and recreate
    """
    print("\n🗄️  STEP 1: Creating Document VectorStore (ChromaDB)")
    print("=" * 70)

    # Initialize document store
    doc_store = DocumentVectorStore(
        persist_directory=persist_dir,
        collection_name=collection_name,
        chunk_size=500,
        chunk_overlap=50
    )

    # Check if collection exists
    if not force_recreate:
        if doc_store.load_existing():
            stats = doc_store.get_stats()
            print(f"\n✅ Using existing ChromaDB collection:")
            print(f"   Collection: {stats['collection_name']}")
            print(f"   Documents: {stats['document_count']}")
            print(f"   Location: {stats['persist_directory']}")
            return doc_store

    # Load documents
    print("\n📥 Loading source data...")

    # Load FAQ as documents
    faq_docs = load_faq_as_documents(here("data/taskflow_faq.csv"))

    # Load main documentation
    main_docs = load_documentation(here("data/taskflow_docs.txt"))

    # Combine all documents
    all_documents = faq_docs + main_docs

    print(f"\n📊 Total documents to process: {len(all_documents)}")
    print(f"   - FAQ entries: {len(faq_docs)}")
    print(f"   - Documentation: {len(main_docs)}")

    # Create vectorstore
    print("\n🔨 Creating ChromaDB vectorstore...")
    doc_store.create_from_documents(
        all_documents, clear_existing=force_recreate)

    # Verify creation
    stats = doc_store.get_stats()
    print(f"\n✅ ChromaDB vectorstore created successfully!")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Total chunks: {stats['document_count']}")
    print(f"   Location: {stats['persist_directory']}")

    return doc_store


def prepare_semantic_cache(cache_file: str = "taskflow_cache_seed.csv"):
    """
    Prepare the initial semantic cache from seed data.
    """
    print("\n💾 STEP 2: Preparing Semantic Cache")
    print("=" * 70)

    # Load seed cache
    print(f"\n📥 Loading cache seed from {cache_file}...")

    cache = SemanticCache.load_from_file(cache_file, distance_threshold=0.35)
    cache_pairs = cache.get_all_pairs()

    print(f"  ✓ Loaded {len(cache_pairs)} Q&A pairs")

    # Show sample entries
    print("\n📋 Sample cache entries:")
    for i, (q, a) in enumerate(cache_pairs[:3], 1):
        print(f"\n{i}. Q: {q}")
        print(f"   A: {a[:100]}{'...' if len(a) > 100 else ''}")

    print(f"\n✅ Semantic cache prepared with {len(cache_pairs)} pairs")

    return cache


def verify_setup(doc_store: DocumentVectorStore):
    """
    Verify the setup by testing retrieval.
    """
    print("\n🧪 STEP 3: Verifying Setup")
    print("=" * 70)

    # Test queries
    test_queries = [
        "How do I create a project?",
        "What are the pricing plans?",
        "Can I use TaskFlow offline?"
    ]

    print("\n🔍 Testing document retrieval...")

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = doc_store.similarity_search(query, k=2)

        if results:
            print(f"  ✓ Found {len(results)} relevant documents")
            print(f"    Top result: {results[0].page_content[:100]}...")
        else:
            print("  ✗ No results found")

    print("\n✅ Retrieval verification complete!")


def show_summary():
    """
    Display summary and next steps.
    """
    print("\n" + "=" * 70)
    print("🎉 DATA PREPARATION COMPLETE!")
    print("=" * 70)

    print("\n📊 What was created:")
    print("   ✓ ChromaDB vectorstore in ./chroma_db/")
    print("   ✓ Persistent document embeddings")
    print("   ✓ Semantic cache prepared")

    print("\n📁 Data files used:")
    print("   • taskflow_faq.csv (30 FAQ entries)")
    print("   • taskflow_docs.txt (main documentation)")
    print("   • taskflow_cache_seed.csv (8 seed cache pairs)")

    print("\n🚀 Next steps:")
    print("   1. Run the Streamlit app:")
    print("      streamlit run streamlit_app.py")
    print("\n   2. Or test with the demo:")
    print("      python demo_chatbot.py")
    print("\n   3. The ChromaDB will persist between runs!")

    print("\n💡 Tips:")
    print("   • ChromaDB is stored in ./chroma_db/")
    print("   • Cache data is in CSV files")
    print("   • Re-run with --force to recreate vectorstore")

    print("\n" + "=" * 70)


def main():
    """Main data preparation workflow."""

    parser = argparse.ArgumentParser(
        description="Prepare data for TaskFlow RAG Chatbot")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate vectorstore even if it exists"
    )
    parser.add_argument(
        "--persist-dir",
        default=str(here("data/chroma_db")),
        help="Directory for ChromaDB persistence (default: data/chroma_db)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TASKFLOW RAG CHATBOT - DATA PREPARATION WITH CHROMADB")
    print("=" * 70)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY environment variable not set!")
        print("Please set it before running this script:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return False

    print("\n✅ OpenAI API key found")

    try:
        # Step 1: Create document vectorstore
        doc_store = create_document_vectorstore(
            persist_dir=args.persist_dir,
            collection_name="taskflow_docs",
            force_recreate=args.force
        )

        # Step 2: Prepare semantic cache
        cache = prepare_semantic_cache(here("data/taskflow_cache_seed.csv"))

        # Step 3: Verify setup
        verify_setup(doc_store)

        # Show summary
        show_summary()

        return True

    except Exception as e:
        print(f"\n❌ Error during data preparation: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
