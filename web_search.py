from tavily import TavilyClient
from typing import List, Dict, Any, Optional
from langchain.schema import Document

class WebSearch:
    """
    Class for handling web search for general skincare queries using Tavily API
    """
    def __init__(self, api_key: str):
        """
        Initialize the web search client
        
        Args:
            api_key: Tavily API key
        """
        self.client = TavilyClient(api_key=api_key)
    
    def search(self, query: str, max_results: int = 5) -> List[Document]:
        """
        Search the web for information related to the query
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of Document objects with search results
        """
        # Add skincare context to the query
        skincare_query = f"skincare {query}"
        
        # Execute search
        search_results = self.client.search(
            query=skincare_query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_raw_content=True,
            include_images=False
        )
        
        # Convert to Document format for consistency with vector store
        documents = []
        if "results" in search_results:
            for result in search_results["results"]:
                # Create document with content and metadata
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "source": "web_search"
                    }
                )
                documents.append(doc)
        
        # Also include the generated answer if available
        if "answer" in search_results and search_results["answer"]:
            answer_doc = Document(
                page_content=search_results["answer"],
                metadata={
                    "title": "Tavily Generated Answer",
                    "source": "web_search_answer"
                }
            )
            # Put the answer at the front of the list
            documents.insert(0, answer_doc)
        
        return documents 