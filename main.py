from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import numpy as np

# Load dataset
loader = TextLoader("data/massachusetts_facts.txt")
docs = loader.load()

# Split text into chunks
# Split the documents into smaller chunks for better processing
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# Actually split the document into chunks
chunks = splitter.split_documents(docs)

# Create local embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings explicitly for inspection
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_vectors = embeddings.embed_documents(chunk_texts)

print("\nEmbeddings for each chunk (showing first 10 dimensions):")
for i, vec in enumerate(chunk_vectors):
    print(f"\nChunk #{i} text:\n{chunk_texts[i]}")
    print(f"Embedding vector (first 10 dims):\n{vec[:10]}")

# Store in FAISS vector DB
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Use local Ollama LLM
llm = Ollama(model="mistral", temperature=0.1)

# Define the prompt template
template = """
You are given the following context to answer the question.
If the answer cannot be found in the context, respond with "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# Build RAG chain with a custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    # Use the custom prompt template
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# Interactive loop for continuous querying until user exits
while True:
    query = input("\nAsk your question about Massachusetts (or type 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Print retrieved chunks
    retrieved_docs = retriever.get_relevant_documents(query)
    print("\nRetrieved Chunks:")
    for i, doc in enumerate(retrieved_docs):
        print(f"Chunk #{i} content:\n{doc.page_content}\n")

    result = qa_chain({"query": query})
    print("A:", result["result"])

    # Print the retrieved chunks that were actually used for the answer
    print("\nSource Chunks used for answer:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"Source #{i} content:\n{doc.page_content}\n")




