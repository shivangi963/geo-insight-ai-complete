import os
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
import io
import base64
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseVectorDB:
    
    
    def __init__(self):
       
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        
        self.client: Client = create_client(supabase_url, supabase_key)
        self.table_name = "property_embeddings"
        
    
        self._initialize_table()
    
    def _initialize_table(self):
   
        try:
          
            self.client.table(self.table_name).select("*").limit(1).execute()
        except Exception:
            print(f"Table {self.table_name} doesn't exist yet. Please create it in Supabase dashboard.")
            print("""
            SQL to create table:
            CREATE TABLE property_embeddings (
                id SERIAL PRIMARY KEY,
                property_id VARCHAR(255) UNIQUE,
                address TEXT,
                image_url TEXT,
                embedding vector(512),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
            
            Also create vector index:
            CREATE INDEX property_embeddings_embedding_idx 
            ON property_embeddings 
            USING ivfflat (embedding vector_cosine_ops);
            """)
    
    def image_to_embedding(self, image_path: str) -> Optional[List[float]]:
        
        try:
         
            with Image.open(image_path) as img:
              
                img = img.resize((224, 224))
                
        
                img_array = np.array(img)
                
              
                embedding = np.random.randn(512).tolist()
                
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding.tolist()
                
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return None
    
    def store_property_embedding(
        self,
        property_id: str,
        address: str,
        image_path: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        try:
           
            embedding = self.image_to_embedding(image_path)
            if not embedding:
                return False
            
 
            data = {
                "property_id": property_id,
                "address": address,
                "embedding": embedding,
                "metadata": metadata or {}
            }
            
           
            response = self.client.table(self.table_name).insert(data).execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            print(f"Error storing embedding: {e}")
            return False
    
    def find_similar_properties(
        self,
        image_path: str,
        limit: int = 5,
        threshold: float = 0.8
    ) -> List[Dict]:
        
        try:
            
            query_embedding = self.image_to_embedding(image_path)
            if not query_embedding:
                return []
            
            response = self.client.table(self.table_name).select("*").execute()
            all_properties = response.data
         
            similar_properties = []
            
            for prop in all_properties:
                if "embedding" in prop:
                 
                    vec1 = np.array(query_embedding)
                    vec2 = np.array(prop["embedding"])
                    
                    similarity = np.dot(vec1, vec2) / (
                        np.linalg.norm(vec1) * np.linalg.norm(vec2)
                    )
                    
                    if similarity >= threshold:
                        prop["similarity"] = float(similarity)
                        similar_properties.append(prop)
            
            similar_properties.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similar_properties[:limit]
            
        except Exception as e:
            print(f"Error finding similar properties: {e}")
            return []
    
    def get_property_by_id(self, property_id: str) -> Optional[Dict]:
        
        try:
            response = self.client.table(self.table_name).select("*").eq(
                "property_id", property_id
            ).execute()
            
            if response.data:
                return response.data[0]
            return None
            
        except Exception as e:
            print(f"Error getting property: {e}")
            return None


vector_db = SupabaseVectorDB()