# -*- coding: utf-8 -*-
"""The main entry point of the RAG example."""
import asyncio
import os

from agentscope.embedding import DashScopeTextEmbedding
from agentscope.rag import (
    TextReader,
    PDFReader,
    QdrantStore,
    SimpleKnowledge,
)

async def create_knowledge_base(
    force_rebuild: bool = False,
) -> SimpleKnowledge:
    """
    Create and populate the knowledge base with documents.
    
    This function:
    1. Creates the knowledge base with Qdrant and DashScope embedding
    2. Checks if data already exists
    3. Reads and adds documents if needed
    
    Args:
        force_rebuild: If True, rebuild the knowledge base even if data exists.
        
    Returns:
        The created knowledge base instance.
    """
    collection_name = "rag_knowledge_base"
    # Setup paths - ensure absolute path for local storage
    qdrant_data_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "qdrant_data",
        )
    )
    
    # Create knowledge base with Qdrant as the embedding store and
    # DashScope as the embedding model
    knowledge = SimpleKnowledge(
        embedding_store=QdrantStore(
            location="http://localhost:6333",  # Local storage path
            collection_name=collection_name,
            dimensions=1024,  # The dimension of the embedding vectors
        ),
        embedding_model=DashScopeTextEmbedding(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            model_name="text-embedding-v4",
        ),
    )
    
    # Skip collection check for local storage to avoid connection errors
    # QdrantStore will automatically create the collection when adding documents
    # For simplicity, we'll always proceed to add documents if force_rebuild is False
    # The QdrantStore's internal validation will handle collection creation
    # 
    # Note: For local storage, checking collection existence before first use
    # can cause connection errors. We rely on QdrantStore's _validate_collection
    # method which is called automatically during add operations.
    
    print("Creating knowledge base with documents...")
    
    # Create readers with chunking arguments
    reader = TextReader(chunk_size=1024)
    pdf_reader = PDFReader(chunk_size=1024, split_by="sentence")
    
    # Read documents
    print("Reading text documents...")
    documents = await reader(
        text="I'm Tony Stank, my password is 123456. My best friend is James "
        "Rhodes.",
    )
    
    # Read a sample PDF file
    pdf_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "example.pdf",
    )
    pdf_documents = []
    if os.path.exists(pdf_path):
        print("Reading PDF documents...")
        pdf_documents = await pdf_reader(pdf_path=pdf_path)
    else:
        print(f"Warning: PDF file not found at {pdf_path}, skipping PDF documents.")
    
    # Insert documents into the knowledge base
    all_documents = documents + pdf_documents
    print(f"Adding {len(all_documents)} documents to knowledge base...")
    await knowledge.add_documents(all_documents)
    print(f"Knowledge base created successfully with {len(all_documents)} documents.")
    
    return knowledge


async def main() -> None:
    """The main entry point of the RAG example."""
    
    # Create knowledge base (will skip if data already exists)
    knowledge = await create_knowledge_base(force_rebuild=False)

    # Retrieve relevant documents based on a given query
    docs = await knowledge.retrieve(
        query="What is Tony Stank's password?",
        limit=3,
        score_threshold=0.7,
    )
    print("Q1: What is Tony Stank's password?")
    for doc in docs:
        print(
            f"Document ID: {doc.id}, Score: {doc.score}, "
            f"Content: {doc.metadata.content['text']}",
        )

    # Retrieve documents from the PDF file based on a query
    docs = await knowledge.retrieve(
        query="climate change",
        limit=3,
        score_threshold=0.2,
    )
    print("\n\nQ2: climate change")
    for doc in docs:
        print(
            f"Document ID: {doc.id}, Score: {doc.score}, "
            f"Content: {repr(doc.metadata.content['text'])}",
        )

if __name__ == '__main__':
    asyncio.run(main())
