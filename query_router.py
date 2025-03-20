import re
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, Any, Literal, Tuple

class QueryRouter:
    """
    Class for handling query routing between product search and general skincare advice
    """
    def __init__(self, openai_api_key: str):
        """
        Initialize the query router
        
        Args:
            openai_api_key: OpenAI API key
        """
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_api_key
        )
        
        # Define the prompt for query classification
        self.classification_prompt = PromptTemplate(
            template="""You are a skincare expert assistant that helps route user queries.
Your goal is to determine if a query is asking for:
1. Product recommendations (user wants specific skincare product suggestions)
2. General skincare advice (user wants information about skincare topics, routines, or ingredients)

Query types:
- PRODUCT: Queries about finding, comparing, or buying specific skincare products
- GENERAL: Queries about skincare topics, routines, treatments, or ingredients without specifically requesting product recommendations

Examples:
- "Recommend a moisturizer for sensitive skin" -> PRODUCT
- "What's a good sunscreen for oily skin?" -> PRODUCT
- "How to treat acne scars" -> GENERAL
- "Benefits of niacinamide in skincare" -> GENERAL
- "What products help with hyperpigmentation?" -> PRODUCT
- "How should I layer skincare products?" -> GENERAL
- "Cleanser under 500 for dry skin" -> PRODUCT
- "How often should I exfoliate my face?" -> GENERAL

User Query: {query}

Analyze this query and respond ONLY with either "PRODUCT" or "GENERAL".""",
            input_variables=["query"]
        )
        
        # Chain for query classification
        self.classification_chain = LLMChain(
            llm=self.llm,
            prompt=self.classification_prompt
        )

        # Define the prompt for extracting price filters
        self.price_filter_prompt = PromptTemplate(
            template="""Extract the maximum price mentioned in this skincare product query.
If a specific price limit is mentioned (like "under 500", "less than 1000", etc.), identify that value.
If a price range is mentioned (like "between 500 and 1000"), identify the maximum value.
If no price is mentioned, respond with "None".

Examples:
- "Moisturizer under 500 rupees" -> 500
- "Face wash less than 1000" -> 1000
- "Products between 500 and 2000" -> 2000
- "Affordable sunscreen" -> None (no specific price mentioned)
- "Serums under Rs. 1200" -> 1200
- "Budget-friendly face cream" -> None (no specific price mentioned)

User Query: {query}

Extract maximum price (respond ONLY with the number or "None"):""",
            input_variables=["query"]
        )
        
        # Chain for price extraction
        self.price_filter_chain = LLMChain(
            llm=self.llm,
            prompt=self.price_filter_prompt
        )
        
    def classify_query(self, query: str) -> Literal["PRODUCT", "GENERAL"]:
        """
        Classify a query as either product recommendation or general skincare advice
        
        Args:
            query: User query text
            
        Returns:
            Classification as either "PRODUCT" or "GENERAL"
        """
        result = self.classification_chain.run(query).strip().upper()
        if result == "PRODUCT":
            return "PRODUCT"
        else:
            return "GENERAL"
    
    def extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract filters from a product query
        
        Args:
            query: User query text
            
        Returns:
            Dictionary of extracted filters
        """
        filters = {}
        
        # Extract price information using LLM
        price_response = self.price_filter_chain.run(query).strip()
        if price_response.lower() != "none" and price_response.isdigit():
            filters["price_max"] = int(price_response)
        
        # Check for "on sale" mentions
        on_sale_pattern = re.compile(r'\b(on\s*sale|discount)\b', re.IGNORECASE)
        if on_sale_pattern.search(query):
            filters["on_sale"] = True
        
        return filters
    
    def parse_query(self, query: str) -> Tuple[Literal["PRODUCT", "GENERAL"], Dict[str, Any]]:
        """
        Parse a query to determine its type and extract any filters
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (query_type, filters)
        """
        query_type = self.classify_query(query)
        filters = {}
        
        if query_type == "PRODUCT":
            filters = self.extract_filters(query)
        
        return query_type, filters 