import os
import weaviate
from weaviate.classes.config import Property, DataType
from langchain_weaviate import WeaviateVectorStore
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import tqdm
import random
import numpy as np

# Create a custom embeddings class using SentenceTransformer
class BGEEmbeddings(Embeddings):
    """Wrapper around BGE embeddings from SentenceTransformer."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en"):
        """Initialize the BGE embeddings.
        
        Args:
            model_name: Name of the BGE model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        except Exception as e:
            print(f"Error loading BGE model: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using BGE.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embeddings, one per input text
        """
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating embeddings with BGE: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using BGE.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding for the query text
        """
        try:
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating query embedding with BGE: {e}")
            raise

class VectorStore:
    """
    Class for handling vector database operations with Weaviate using BGE embeddings
    """
    def __init__(self, cluster_url: str, api_key: str):
        """
        Initialize the vector store with Weaviate credentials
        
        Args:
            cluster_url: Weaviate cluster URL
            api_key: Weaviate API key
        """
        self.cluster_url = cluster_url
        self.api_key = api_key
        self.client = None
        self.index_name = "SkinCareProducts"
        # Initialize BGE embeddings
        self.embeddings = BGEEmbeddings()
        self.vector_store = None
        
    def connect(self) -> None:
        """
        Connect to Weaviate cluster
        """
        # Connect to Weaviate cloud using the latest API
        try:
            # Connect without any additional headers
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.cluster_url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key=self.api_key)
            )
            print("Connected to Weaviate cluster")
        except Exception as e:
            print(f"Error connecting to Weaviate: {e}")
            raise
    
    def create_schema(self) -> None:
        """
        Create schema in Weaviate for skincare products
        """
        if self.client is None:
            self.connect()
        
        # Check if collection already exists
        try:
            # In Weaviate v4.x, we use collections instead of schema.get
            if self.client.collections.exists(self.index_name):
                print(f"Collection '{self.index_name}' already exists")
                return
        except Exception as e:
            print(f"Error checking if collection exists: {e}")
            # Continue with creation attempt
        
        # Create collection with properties
        try:
            # In Weaviate v4.x, we create collections directly
            collection = self.client.collections.create(
                name=self.index_name,
                description="Skincare product information for RAG system",
                # Use 'none' vectorizer since we'll provide vectors manually from BGE
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                vector_index_config=weaviate.classes.config.Configure.VectorIndex.hnsw(
                    distance_metric=weaviate.classes.config.VectorDistances.COSINE
                ),
                properties=[
                    weaviate.classes.config.Property(
                        name="text",
                        data_type=weaviate.classes.config.DataType.TEXT,
                    ),
                    weaviate.classes.config.Property(
                        name="product_name",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="description",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="type",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="tags",
                        data_type=weaviate.classes.config.DataType.TEXT_ARRAY
                    ),
                    weaviate.classes.config.Property(
                        name="category",
                        data_type=weaviate.classes.config.DataType.TEXT
                    ),
                    weaviate.classes.config.Property(
                        name="price",
                        data_type=weaviate.classes.config.DataType.NUMBER
                    ),
                    weaviate.classes.config.Property(
                        name="key_ingredients",
                        data_type=weaviate.classes.config.DataType.TEXT_ARRAY
                    ),
                    weaviate.classes.config.Property(
                        name="on_sale",
                        data_type=weaviate.classes.config.DataType.TEXT_ARRAY
                    )
                ]
            )
            print(f"Created collection '{self.index_name}' in Weaviate")
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
    def reset_collection(self) -> None:
        """
        Delete and recreate the collection (use with caution)
        """
        if self.client is None:
            self.connect()
            
        try:
            # Delete the collection if it exists
            if self.client.collections.exists(self.index_name):
                self.client.collections.delete(self.index_name)
                print(f"Deleted existing collection '{self.index_name}'")
            
            # Recreate the collection
            self.create_schema()
            print(f"Reset of collection '{self.index_name}' completed")
        except Exception as e:
            print(f"Error resetting collection: {e}")
            raise
    
    def _create_random_vector(self, seed=None):
        """
        Create a random vector for demonstration purposes
        
        Args:
            seed: Optional seed for reproducibility
            
        Returns:
            A random 1024-dimensional vector (matching BGE embedding dimensions)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create a random vector with 1024 dimensions (matching BGE embedding size)
        vector = np.random.rand(1024)
        # Normalize to unit length for cosine similarity
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()
    
    def load_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Load documents into Weaviate
        
        Args:
            documents: List of document dictionaries with text and metadata
        """
        if self.client is None:
            self.connect()
            
        # Reset the collection to make sure we have a clean state with correct settings
        try:
            self.reset_collection()
        except Exception as e:
            print(f"Warning: Collection reset failed but continuing: {e}")
        
        try:
            # Make sure the collection exists
            if not self.client.collections.exists(self.index_name):
                self.create_schema()
                
            # Get the collection and add documents directly using batch import
            print("Adding documents directly to Weaviate...")
            collection = self.client.collections.get(self.index_name)
            
            # Batch import data
            with collection.batch.dynamic() as batch:
                for i, doc in enumerate(documents):
                    if i % 100 == 0:
                        print(f"Processing document {i}/{len(documents)}")
                        
                    # Combine text and metadata into properties
                    properties = {
                        "text": doc["text"],
                    }
                    
                    # Add all metadata properties
                    for key, value in doc["metadata"].items():
                        properties[key] = value
                    
                    # Generate embedding for the document text
                    vector = self.embeddings.embed_query(doc["text"])
                    
                    # Add the object with the embedding vector
                    batch.add_object(properties=properties, vector=vector)
            
            print(f"Successfully added {len(documents)} documents to Weaviate")
            
            # Initialize vector store for retrieval
            self.vector_store = WeaviateVectorStore(
                client=self.client,
                index_name=self.index_name,
                text_key="text",
                embedding=self.embeddings
            )
            
        except Exception as e:
            print(f"Error loading documents into Weaviate: {e}")
            raise
    
    def delete_class(self) -> None:
        """
        Delete the class from Weaviate (use with caution)
        """
        if self.client is None:
            self.connect()
        
        try:
            # In Weaviate v4.x, we delete collections
            if self.client.collections.exists(self.index_name):
                self.client.collections.delete(self.index_name)
                print(f"Deleted collection '{self.index_name}' from Weaviate")
            else:
                print(f"Collection '{self.index_name}' does not exist")
        except Exception as e:
            print(f"Error deleting collection: {e}")
            raise
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Perform similarity search in vector store
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        if self.vector_store is None:
            print("Vector store not initialized. Attempting to initialize...")
            try:
                # Initialize vector store for search if not already done
                self.vector_store = WeaviateVectorStore(
                    client=self.client,
                    index_name=self.index_name,
                    text_key="text",
                    embedding=self.embeddings
                )
                print("Vector store initialized successfully")
            except Exception as e:
                print(f"Error initializing vector store: {e}")
                # Return empty list if we can't initialize
                return []
        
        try:
            print(f"Searching for query: '{query}'")
            collection = self.client.collections.get(self.index_name)
            
            # Generate embedding for the query using BGE
            query_vector = self.embeddings.embed_query(query)
            
            # Perform vector search
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
            )
            
            # Convert to Document format for consistency
            results = []
            for item in response.objects:
                props = item.properties
                doc = Document(
                    page_content=props.get("text", ""),
                    metadata={
                        k: v for k, v in props.items() if k != "text"
                    }
                )
                results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            
            # Fall back to keyword search by filtering on text content
            try:
                print("Falling back to keyword search...")
                collection = self.client.collections.get(self.index_name)
                response = collection.query.fetch_objects(
                    filters=weaviate.classes.query.Filter.by_property("text").contains(query),
                    limit=top_k
                )
                
                # Convert to Document format
                results = []
                for item in response.objects:
                    props = item.properties
                    doc = Document(
                        page_content=props.get("text", ""),
                        metadata={
                            k: v for k, v in props.items() if k != "text"
                        }
                    )
                    results.append(doc)
                
                return results
            except Exception as e2:
                print(f"Keyword search also failed: {e2}")
                return []
    
    def hybrid_search(self, 
                     query: str, 
                     alpha: float = 0.5,
                     top_k: int = 5) -> List[Document]:
        """
        Hybrid search combining keyword and vector search
        
        Args:
            query: Query text
            alpha: Weight between keyword (0) and vector (1) search
            top_k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        if self.client is None:
            self.connect()
        
        try:
            print(f"Performing hybrid search for: '{query}' with alpha={alpha}")
            collection = self.client.collections.get(self.index_name)
            
            # Generate embedding for the query using BGE
            query_vector = self.embeddings.embed_query(query)
            
            # Use Weaviate's hybrid search API
            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=top_k
            )
            
            # Convert to Document format
            results = []
            for item in response.objects:
                props = item.properties
                doc = Document(
                    page_content=props.get("text", ""),
                    metadata={
                        k: v for k, v in props.items() if k != "text"
                    }
                )
                results.append(doc)
            
            return results
        
        except Exception as e:
            print(f"Error during hybrid search: {e}")
            
            # Fall back to regular similarity search
            try:
                return self.similarity_search(query, top_k=top_k)
            except Exception as e2:
                print(f"Fallback search also failed: {e2}")
                return []
    
    def filter_search(self, 
                      query: str, 
                      filters: Dict[str, Any],
                      top_k: int = 5) -> List[Document]:
        """
        Perform search with metadata filters
        
        Args:
            query: Query text
            filters: Dictionary of metadata filters
            top_k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        if self.client is None:
            self.connect()
        
        try:
            print(f"Filtering search for query: '{query}' with filters: {filters}")
            collection = self.client.collections.get(self.index_name)
            
            # Start with base filter
            weaviate_filter = None
            
            # Add price filter if specified
            if "price_max" in filters:
                price_value = float(filters["price_max"])
                weaviate_filter = weaviate.classes.query.Filter.by_property("price").less_than_equal(price_value)
            
            # Add on_sale filter if specified
            elif "on_sale" in filters and filters["on_sale"]:
                # For array properties, we need to check if the array contains any of the values
                weaviate_filter = weaviate.classes.query.Filter.by_property("on_sale").contains_any(["On Sale"])
            
            # Generate embedding for the query using BGE
            query_vector = self.embeddings.embed_query(query)
            
            # Combine vector search with filter
            if weaviate_filter:
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    filters=weaviate_filter,
                    limit=top_k
                )
            else:
                # No filter, just do vector search
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=top_k
                )
            
            # Convert to Document format
            results = []
            for item in response.objects:
                props = item.properties
                doc = Document(
                    page_content=props.get("text", ""),
                    metadata={
                        k: v for k, v in props.items() if k != "text"
                    }
                )
                results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"Error during filtered search: {e}")
            
            # Fall back to regular search
            try:
                return self.similarity_search(query, top_k=top_k)
            except Exception as e2:
                print(f"Fallback search also failed: {e2}")
                return [] 