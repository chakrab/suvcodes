import asyncio
import chromadb
from dotenv import load_dotenv
from openai import AsyncOpenAI

class BasicRAGAgent:
    """
    A basic Retrieval-Augmented Generation (RAG) agent that can embed text chunks
    and retrieve similar chunks based on a query using ChromaDB and an OpenAI-compatible
    embedding model.
    """
    def __init__(self):
        load_dotenv()

        endpoint = "http://localhost:11434/v1"
        self.model_alias = "nomic-embed-text:v1.5"
        self.embed_client = AsyncOpenAI(base_url=endpoint)

        #persist_dir = "./book_embeds.db"
        #self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self.chroma_client.get_or_create_collection(name="book_embeddings")

    """
    Splits text into chunks of specified size with overlap.
    """
    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    """
    Reads a file and splits its content into chunks.
    """
    def split_file_in_chunks(self, file_path: str, name: str, chunk_size: int = 500, overlap: int = 50) -> dict:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return {"name": name, "chunks": self.split_text(text, chunk_size, overlap)}
    
    """
    Asynchronously gets embeddings for a given text chunk.
    """
    async def get_embeddings(self, chunk):
        response = await self.embed_client.embeddings.create(
            model=self.model_alias,
            input=chunk
        )
        return response.data[0].embedding

    """
    Asynchronously embeds a list of text chunks and stores them in the ChromaDB collection.
    """
    async def embed_text(self, lst_file_chunks: str):
        for chunk in lst_file_chunks:
            total_chunks = len(chunk.get("chunks"))
            for idx, text_chunk in enumerate(chunk.get("chunks")):
                print(f"Embedding: {chunk.get('name')} - Chunk {idx + 1} of {total_chunks}")
                this_chunk = await self.get_embeddings(text_chunk)
                self.collection.add(
                    documents=[text_chunk],
                    metadatas=[{"source": chunk.get("name")}], 
                    embeddings=this_chunk, 
                    ids=str(chunk.get('name') + "_" + str(idx))
                )
    
    """
    Asynchronously retrieves similar text chunks based on a query.
    """
    async def retrieve_similar_chunks(self, query: str, n_results: int = 5):
        query_embedding = await self.get_embeddings(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

if __name__ == "__main__":
    # All books are from Project Gutenberg, https://www.gutenberg.org/. Locally saved for convenience.
    files_to_chunk = [
        {"file_path": "doc/Just_so_Stories.txt", "name": "Just so Stories"},
        {"file_path": "doc/The_House_at_Pooh_Corner.txt", "name": "The House at Pooh Corner"},
        {"file_path": "doc/Wind_in_the_Willow.txt", "name": "Wind in the Willows"},
    ]
    agent = BasicRAGAgent()
    chunks = []
    for file in files_to_chunk:
        chunked_file = agent.split_file_in_chunks(file["file_path"], file["name"])
        chunks.append(chunked_file)
    asyncio.run(agent.embed_text(chunks))
    query = "What poetry did Pooh come up with about fir-cones?"
    """
    Here is a myst'ry
    About a little fir-tree.
    Owl says it's _his_ tree,
    And Kanga says it's _her_ tree.
    """
    results = asyncio.run(agent.retrieve_similar_chunks(query, n_results=2))
    print("Retrieved Chunks:")
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        print(f"Source: {metadata['source']}\nContent: {doc}\n")
