import pandas as pd
import json
import ast
from typing import Dict, List, Any, Optional
import numpy as np

class DataProcessor:
    """
    Class for processing skincare product data from XLSX file
    """
    def __init__(self, file_path: str):
        """
        Initialize the DataProcessor with the path to the XLSX file
        
        Args:
            file_path: Path to the XLSX file containing product data
        """
        self.file_path = file_path
        self.df = None
        self.processed_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from XLSX file
        
        Returns:
            Pandas DataFrame containing the loaded data
        """
        self.df = pd.read_excel(self.file_path)
        print(f"Loaded {len(self.df)} products")
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the loaded data
        
        Returns:
            Cleaned pandas DataFrame
        """
        if self.df is None:
            self.load_data()
        
        # Handle NaN values
        self.df = self.df.fillna('')
        
        # Convert price to numeric
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')
        
        # Process tags and ingredients as lists if they're strings
        for col in ['Tags', 'Key Ingredients', 'On Sale']:
            self.df[col] = self.df[col].apply(self._parse_list_field)
        
        print("Data cleaning completed")
        return self.df
    
    def _parse_list_field(self, field_value: Any) -> List[str]:
        """
        Parse string representations of lists into actual lists
        
        Args:
            field_value: Value to parse, could be string, list or other
            
        Returns:
            List of strings
        """
        if isinstance(field_value, list):
            return field_value
        
        if isinstance(field_value, str) and field_value:
            try:
                # Try to parse as JSON/list literal
                parsed = ast.literal_eval(field_value)
                if isinstance(parsed, list):
                    return parsed
                return [str(parsed)]
            except (SyntaxError, ValueError):
                # If can't parse, return as single item list
                return [field_value]
        
        return [] if field_value == '' else [str(field_value)]
    
    def create_documents(self) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to a list of document dictionaries for vector db
        
        Returns:
            List of document dictionaries
        """
        if self.df is None or len(self.df) == 0:
            self.clean_data()
        
        documents = []
        
        for _, row in self.df.iterrows():
            # Create the text field by combining relevant product info
            # This will be used for embeddings and semantic search
            text_for_embedding = f"""
            Product: {row['Product Name']}
            Description: {row['Description']}
            Type: {row['Type']}
            Category: {row['Category']}
            Key Ingredients: {', '.join(row['Key Ingredients'])}
            """
            
            # Create the document with both text and metadata
            document = {
                "text": text_for_embedding.strip(),
                "metadata": {
                    "product_name": row['Product Name'],
                    "description": row['Description'],
                    "type": row['Type'],
                    "tags": row['Tags'],
                    "category": row['Category'],
                    "price": row['Price'],
                    "key_ingredients": row['Key Ingredients'],
                    "on_sale": row['On Sale']
                }
            }
            
            documents.append(document)
        
        self.processed_data = documents
        print(f"Created {len(documents)} document objects")
        return documents 