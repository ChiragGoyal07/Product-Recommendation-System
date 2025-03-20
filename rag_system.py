import os
from typing import Dict, List, Any, Optional
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from data_processor import DataProcessor
from vector_store import VectorStore
from query_router import QueryRouter
from web_search import WebSearch

class RAGSystem:
    """
    Main class for the Retrieval Augmented Generation system
    """
    def __init__(self, config: Dict[str, str]):
        """
        Initialize the RAG system with configuration
        
        Args:
            config: Dictionary containing API keys and configuration
        """
        # Set up configuration
        self.openai_api_key = config.get("openai_api_key", "")
        self.tavily_api_key = config.get("tavily_api_key", "")
        self.weaviate_cluster_url = config.get("weaviate_cluster_url", "")
        self.weaviate_api_key = config.get("weaviate_api_key", "")
        self.product_data_path = config.get("product_data_path", "")
        
        # Initialize components
        self.data_processor = DataProcessor(self.product_data_path)
        self.vector_store = VectorStore(self.weaviate_cluster_url, self.weaviate_api_key)
        self.query_router = QueryRouter(self.openai_api_key)
        self.web_search = WebSearch(self.tavily_api_key)
        
        # Set up LLM with higher temperature for more creative responses
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=self.openai_api_key
        )
        
        # Set up memory with a larger buffer size
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Set up prompts with enhanced context handling
        self.product_prompt = PromptTemplate(
            template="""You are a knowledgeable skincare assistant. Your goal is to provide accurate, helpful product recommendations based on the user's query and conversation history.

CONVERSATION HISTORY:
{chat_history}

CURRENT CONTEXT (Product Information):
{context}

USER QUERY: {question}

Instructions for generating response:
1. First, analyze the conversation history to understand any previously mentioned:
   - Skin type/concerns
   - Product preferences
   - Budget constraints
   - Previously recommended products
   
2. Then, provide a detailed response that:
   - References relevant previous recommendations if they fit the current query
   - Suggests new products that complement previous recommendations
   - Explains why each product is suitable based on the user's specific needs
   - Includes key features and ingredients of recommended products
   - Mentions price information for each product
   
3. If the query references previous recommendations but you don't find relevant context in the history:
   - Focus on providing new, personalized recommendations
   - Explain that you're making fresh recommendations based on the current query
   - Ask clarifying questions if needed

4. Always prioritize PERSONALIZATION over history if you must choose:
   - Give precedence to the current query's specific needs
   - Use history to enhance, not restrict, your recommendations
   - Feel free to suggest different products if they better match current needs

Answer in a friendly, conversational tone. Be specific and reference actual products from the context.
If you're not sure about something, be honest and suggest consulting with a dermatologist for personalized advice.""",
            input_variables=["context", "chat_history", "question"]
        )
        
        self.general_prompt = PromptTemplate(
            template="""You are a knowledgeable skincare assistant. Your goal is to provide helpful, educational information about skincare based on the user's query and conversation history.

CONVERSATION HISTORY:
{chat_history}

CURRENT CONTEXT (Search Results):
{context}

USER QUERY: {question}

Instructions for generating response:
1. First, analyze the conversation history to understand:
   - Previously discussed skincare concerns
   - Mentioned routines or practices
   - Specific products or ingredients discussed
   
2. Then, provide a detailed response that:
   - Builds upon previous advice if relevant
   - Provides new, evidence-based information
   - Explains skincare concepts clearly
   - Offers practical, actionable advice
   
3. If the query references previous discussion but you don't find relevant context:
   - Focus on providing fresh, comprehensive advice
   - Explain that you're offering new recommendations
   - Ask clarifying questions if needed

4. Always prioritize ACCURACY and PERSONALIZATION over history:
   - Focus on providing the most accurate, up-to-date information
   - Tailor advice to the current query's specific needs
   - Use history to enhance, not limit, your response

Answer in a friendly, conversational tone. Provide explanations that are easy to understand but scientifically accurate.
If there are conflicting opinions in the search results, acknowledge them and provide balanced information.
If you're not sure about something, be honest and suggest consulting with a dermatologist.""",
            input_variables=["context", "chat_history", "question"]
        )
        
        # Retrieval chains will be initialized later
        self.product_chain = None
        self.general_chain = None
    
    def initialize(self) -> None:
        """
        Initialize the RAG system by setting up the vector store and retrieval chains
        """
        # Connect to the vector database
        self.vector_store.connect()
        
        # Don't initialize retrieval chains yet - we'll do that after loading documents
        print("RAG system initialized successfully")
    
    def _init_retrieval_chains(self) -> None:
        """
        Initialize the retrieval chains for both product and general queries
        """
        # Make sure vector_store is initialized
        if self.vector_store.vector_store is None:
            print("Warning: Vector store not initialized with documents yet. Retrieval chains will be initialized after loading documents.")
            return
            
        # Set up the product retrieval chain with memory
        self.product_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.product_prompt},
            return_source_documents=True,
            chain_type="stuff"
        )
        
        print("Retrieval chains initialized successfully")
    
    def process_data(self) -> None:
        """
        Process the product data and load it into the vector store
        """
        # Load and clean data
        self.data_processor.load_data()
        self.data_processor.clean_data()
        
        # Create documents for vector database
        documents = self.data_processor.create_documents()
        
        # Create schema and load documents into vector store
        self.vector_store.create_schema()
        self.vector_store.load_documents(documents)
        
        # Now that documents are loaded, initialize the retrieval chains
        self._init_retrieval_chains()
        
        print("Data processing and loading completed")
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and generate a response
        
        Args:
            query: User query text
            
        Returns:
            Generated response text
        """
        # Classify the query and extract filters
        query_type, filters = self.query_router.parse_query(query)
        
        if query_type == "PRODUCT":
            # Handle product recommendation query
            return self._process_product_query(query, filters)
        else:
            # Handle general skincare advice query
            return self._process_general_query(query)
    
    def _process_product_query(self, query: str, filters: Dict[str, Any]) -> str:
        """
        Process a product recommendation query
        
        Args:
            query: User query text
            filters: Dictionary of extracted filters
            
        Returns:
            Generated response text
        """
        # Check if vector store is initialized
        if self.vector_store.vector_store is None:
            return "I'm not ready to answer product questions yet. Please process the product data first by clicking 'Process Product Data' in the sidebar."
            
        # Initialize the product chain if not done already
        if self.product_chain is None:
            self._init_retrieval_chains()
            
        # Make sure product chain is initialized
        if self.product_chain is None:
            return "I'm having trouble accessing the product database. Please try processing the data again or contact support."
        
        try:
            # Use filters if available
            if filters:
                # Get relevant documents with filters
                docs = self.vector_store.filter_search(query, filters)
                
                # Create a formatted context from the documents
                context_text = self._format_documents(docs)
                
                # Generate response using the LLM with the filtered results
                response = self.llm.invoke(
                    self.product_prompt.format(
                        context=context_text,
                        chat_history=self.memory.buffer,
                        question=query
                    )
                )
                
                # Update memory manually
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(response.content)
                
                return response.content
            else:
                # Use the standard retrieval chain
                response = self.product_chain({"question": query})
                return response["answer"]
                
        except Exception as e:
            print(f"Error processing product query: {e}")
            return "I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."
    
    def _process_general_query(self, query: str) -> str:
        """
        Process a general skincare advice query
        
        Args:
            query: User query text
            
        Returns:
            Generated response text
        """
        # Perform web search for the query
        search_results = self.web_search.search(query)
        
        # Format the search results
        context_text = self._format_documents(search_results)
        
        # Generate response using the LLM with the search results
        response = self.llm.invoke(
            self.general_prompt.format(
                context=context_text,
                chat_history=self.memory.buffer,
                question=query
            )
        )
        
        # Update memory
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response.content)
        
        return response.content
    
    def _format_documents(self, documents: List[Document]) -> str:
        """
        Format a list of documents into a single context string
        
        Args:
            documents: List of Document objects
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found."
        
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            # Format the document
            if doc.metadata.get("source") == "web_search_answer":
                # Format Tavily generated answer differently
                formatted_docs.append(f"GENERATED ANSWER: {doc.page_content}")
            elif doc.metadata.get("source") == "web_search":
                # Format web search result
                title = doc.metadata.get("title", "Unknown Title")
                url = doc.metadata.get("url", "Unknown URL")
                formatted_docs.append(f"SOURCE {i+1}: {title}\nURL: {url}\n{doc.page_content}")
            else:
                # Format product document
                product_name = doc.metadata.get("product_name", "Unknown Product")
                price = doc.metadata.get("price", "Unknown Price")
                category = doc.metadata.get("category", "Unknown Category")
                key_ingredients = ", ".join(doc.metadata.get("key_ingredients", []))
                
                formatted = f"PRODUCT {i+1}: {product_name}\n"
                formatted += f"PRICE: {price}\n"
                formatted += f"CATEGORY: {category}\n"
                formatted += f"KEY INGREDIENTS: {key_ingredients}\n"
                formatted += f"DESCRIPTION: {doc.metadata.get('description', '')}\n"
                
                formatted_docs.append(formatted)
        
        return "\n\n".join(formatted_docs)
    
    def reset_conversation(self) -> None:
        """
        Reset the conversation history
        """
        self.memory.clear()
        print("Conversation history cleared") 
        