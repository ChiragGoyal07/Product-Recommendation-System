# Product Recommendation RAG System

This project implements a Retrieval Augmented Generation (RAG) system for e product recommendations and general advice. The system intelligently routes queries between product recommendations and general information using a hybrid approach.

## Key Features

- **Hybrid Query Routing**: Automatically distinguishes between product recommendation queries and general questions
- **Vector Database Integration**: Uses Weaviate for efficient similarity search of  products
- **Metadata Filtering**: Allows filtering products by price, sale status, and other attributes
- **Web Search Integration**: Leverages Tavily API for answering general questions
- **Conversation History**: Maintains context of the conversation for natural dialogue
- **User-Friendly UI**: Clean Streamlit interface for easy interaction

## Tech Stack

- **LLM**: OpenAI GPT-4o
- **Embeddings**: Sentence Transformer BAAI/bge-large-en
- **Vector Database**: Weaviate
- **Web Search**: Tavily
- **Orchestration**: LangChain
- **Frontend**: Streamlit
- **Data Processing**: Pandas

## System Architecture

The system is composed of several modular components:

1. **Data Processor**: Handles loading, cleaning, and preparing product data
2. **Vector Store**: Manages interactions with the Weaviate vector database
3. **Query Router**: Classifies queries and extracts relevant filters
4. **Web Search**: Performs web searches for general  questions
5. **RAG System**: Orchestrates the entire pipeline, from query intake to response generation
6. **Streamlit UI**: Provides the user interface for interacting with the system

## Query Flow

1. User submits a query through the Streamlit UI
2. Query Router classifies the query as either a product recommendation or general advice
3. For product queries:
   - Metadata filters are extracted (e.g., price limits)
   - Vector search retrieves relevant products
   - LLM generates personalized recommendations
4. For general queries:
   - Web search retrieves relevant information
   - LLM generates educational responses
5. Response is displayed to the user and conversation history is updated

## Setup and Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Weaviate cloud instance
- Tavily API key

### Environment Variables

Create a `.env` file with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
WEAVIATE_CLUSTER_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key
PRODUCT_DATA_PATH=path_to_product_data.xlsx
```

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install langchain langchain-openai langchain-weaviate langchain-community streamlit weaviate-client tavily-python python-dotenv pandas openpyxl tqdm numpy
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Open the application in your browser
2. Click "Initialize System" in the sidebar
3. Click "Process Product Data" to load the  products into the vector database
4. Start asking questions in the chat interface:
   - Product recommendations: "Recommend a moisturizer for oily skin under 1000"
   - General  advice: "How to treat acne scars?"

## Project Structure

```
├── app.py                 # Streamlit UI
├── data_processor.py      # Data loading and processing
├── vector_store.py        # Weaviate integration
├── query_router.py        # Query classification and filtering
├── web_search.py          # Tavily web search integration
├── rag_system.py          # Main RAG orchestration
├── .env                   # Environment variables
└── Cleaned_Product_Data.xlsx  # Product dataset
```

## Future Improvements

- Implement more advanced filter extraction (skin type, product category, etc.)
- Add product image display in the UI
- Implement reranking for better search relevance
- Add user feedback mechanisms for result improvement
- Expand the product database with more items and attributes 
