import os
import subprocess
import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "langchain", "langchain-openai", "langchain-weaviate", "langchain-community",
        "streamlit", "weaviate-client", "tavily-python", "python-dotenv",
        "pandas", "openpyxl", "tqdm", "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        install = input("Do you want to install them now? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        else:
            print("Please install the missing packages and try again.")
            sys.exit(1)

def check_env_file():
    """Check if .env file exists with required variables"""
    if not os.path.exists('.env'):
        print("No .env file found. Creating one...")
        with open('.env', 'w') as f:
            f.write("OPENAI_API_KEY=\n")
            f.write("TAVILY_API_KEY=\n")
            f.write("WEAVIATE_CLUSTER_URL=\n")
            f.write("WEAVIATE_API_KEY=\n")
            f.write("PRODUCT_DATA_PATH=Cleaned_Product_Data.xlsx\n")
        
        print("\nPlease edit the .env file with your API keys before running the app.")
        sys.exit(1)
    
    # Check if API keys are set
    from dotenv import load_dotenv
    load_dotenv()
    
    missing_vars = []
    for var in ["OPENAI_API_KEY", "TAVILY_API_KEY", "WEAVIATE_CLUSTER_URL", "WEAVIATE_API_KEY"]:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"The following environment variables are missing in .env: {', '.join(missing_vars)}")
        print("Please update the .env file with your API keys before running the app.")
        sys.exit(1)

def run_app():
    """Run the Streamlit app"""
    print("Starting Skincare Product Recommendation RAG System...")
    subprocess.call(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    print("Checking dependencies...")
    check_dependencies()
    
    print("Checking environment variables...")
    check_env_file()
    
    run_app() 