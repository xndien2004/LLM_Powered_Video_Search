from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class LangchainSearch:
    def __init__(self) -> None:
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/stsb-xlm-r-multilingual",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        # Initialize FAISS index with the dimensionality of the embeddings
        embedding_dim = len(self.embeddings.embed_query(""))
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Initialize the vector store
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def add_document(self, documents: list[Document]):
        """Add documents to the vector store."""
        if not isinstance(documents, list):
            raise ValueError("Documents should be a list of Document objects.")
        
        self.vector_store.add_documents(documents=documents)
        print("Documents added")

    def delete_document(self, docstore_id: str):
        """Delete a document from the vector store using its ID."""
        if docstore_id not in self.vector_store.index_to_docstore_id.values():
            raise ValueError("Document ID not found in the docstore.")
        
        self.vector_store.delete_document(docstore_id)
        print(f"Document with ID {docstore_id} deleted")

    def search(self, query: str, k=5, filter_kwargs: dict = None):
        """Search for documents similar to the query."""
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_kwargs
        )
        return results

    def search_with_score(self, query: str, k=5, filter_kwargs: dict = None):
        """Search for documents similar to the query with scores."""
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_kwargs
        )
        return self._format_output(results)

    def retriever(self, query: str, filter_kwargs: dict, search_type: str, search_kwargs: dict):
        """Retrieve documents using specified search type."""
        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        results = retriever.invoke(input=query, filter=filter_kwargs)
        return results

    def save_index(self, path: str):
        """Save the FAISS index to a local file."""
        self.vector_store.save_local(path)
        print(f"Index saved at {path}")

    def load_index(self, path: str):
        """Load a FAISS index from a local file."""
        self.vector_store = FAISS.load_local(
            folder_path=path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Index loaded from {path}")

    @staticmethod
    def _format_output(results):
        """Format the output of a search with scores."""
        flattened_frame_idx = []
        flattened_frame_path = []
        flattened_scores = []

        for document, score in results:
            frame_idx = document.metadata.get('frame_idx', [])
            frame_paths = document.metadata.get('source', [])
            
            # Ensure frame_idx and frame_paths are lists and match in length
            if not isinstance(frame_idx, list):
                frame_idx = [frame_idx]
            if not isinstance(frame_paths, list):
                frame_paths = [frame_paths]
            if len(frame_idx) != len(frame_paths):
                raise ValueError("Frame indices and paths must have the same length.")

            for idx, path in zip(frame_idx, frame_paths):
                flattened_frame_idx.append(idx)
                flattened_frame_path.append(path)
                flattened_scores.append(score)

        return flattened_scores, flattened_frame_idx, flattened_frame_path


def main(): 
    import time
    search_instance = LangchainSearch()
    
    # documents = [Document(page_content="Xin chào", metadata={"source": "1"})]
    # search_instance.add_document(documents)
    
    start = time.time()
    search_instance.load_index("./media/faiss_audio_context_index")
    end = time.time()
    print("Time load index : ", end - start)
    # Perform a search

    results = search_instance.search_with_score("Xin chào")
    print("Time search:", time.time() - end)
    print("Search results:", results)

if __name__ == "__main__":
    main()
