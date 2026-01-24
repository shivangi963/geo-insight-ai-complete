import os
from typing import List, Optional, Dict 
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import traceback

load_dotenv()

# Try to import required libraries
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("⚠️ Supabase not installed. Vector search disabled.")
    SUPABASE_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    print("⚠️ CLIP not installed. Using dummy embeddings.")
    CLIP_AVAILABLE = False


class SupabaseVectorDB:
    """
    FIXED: Supabase Vector Database with real CLIP embeddings
    """
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.table_name = "property_embeddings"
        self.client: Optional[Client] = None
        self.clip_model = None
        self.clip_processor = None
        
        # Validate credentials
        if not self.supabase_url or not self.supabase_key or \
           self.supabase_url == "your_url_here" or self.supabase_key == "your_key_here":
            print("⚠️ Supabase credentials not configured in .env")
            print("   Set SUPABASE_URL and SUPABASE_KEY to enable vector search")
            self.enabled = False
            return
        
        if not SUPABASE_AVAILABLE:
            print("⚠️ Supabase library not available")
            self.enabled = False
            return
        
        try:
            # Initialize Supabase client
            self.client = create_client(self.supabase_url, self.supabase_key)
            print("✅ Supabase client initialized")
            
            # Initialize CLIP model for embeddings
            if CLIP_AVAILABLE:
                self._initialize_clip_model()
            
            self.enabled = True
            
            # Verify table exists
            self._verify_table()
        
        except Exception as e:
            print(f"❌ Supabase initialization failed: {e}")
            self.enabled = False
    
    def _initialize_clip_model(self):
        """Initialize CLIP model for image embeddings"""
        try:
            print("🔄 Loading CLIP model for embeddings...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Set to eval mode
            self.clip_model.eval()
            
            print("✅ CLIP model loaded successfully")
        except Exception as e:
            print(f"⚠️ CLIP model loading failed: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def _verify_table(self):
        
        try:
           
            result = self.client.table(self.table_name).select("*").limit(1).execute()
            print(f"✅ Table '{self.table_name}' exists")
        except Exception as e:
            print(f"⚠️ Table verification failed: {e}")
            print(f"""
            CREATE TABLE IN SUPABASE
            
            Run this SQL in Supabase SQL editor:
            
            -- Enable pgvector extension
            CREATE EXTENSION IF NOT EXISTS vector;
            
            -- Create table
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                property_id VARCHAR(255) UNIQUE,
                address TEXT,
                image_url TEXT,
                embedding vector(512),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
            
            -- Create vector index for faster similarity search
            CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
            ON {self.table_name} 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            
            -- Create index on property_id
            CREATE INDEX IF NOT EXISTS {self.table_name}_property_id_idx
            ON {self.table_name}(property_id);
          
            """)
    
    def image_to_embedding(self, image_path: str) -> Optional[List[float]]:
        """
        FIXED: Generate real CLIP embeddings from image
        """
        if not self.enabled:
            return None
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Use CLIP if available
            if self.clip_model and self.clip_processor:
                with torch.no_grad():
                    inputs = self.clip_processor(images=image, return_tensors="pt")
                    image_features = self.clip_model.get_image_features(**inputs)
                    
                    # Normalize embedding
                    embedding = image_features / image_features.norm(dim=-1, keepdim=True)
                    embedding = embedding.squeeze().cpu().numpy()
                    
                    return embedding.tolist()
            
            else:
                # Fallback: Use simple image features
                print("⚠️ Using fallback embedding method")
                
                # Resize to consistent size
                image = image.resize((224, 224))
                img_array = np.array(image)
                
                # Extract simple features
                # Color histogram
                r_hist = np.histogram(img_array[:,:,0], bins=32, range=(0, 256))[0]
                g_hist = np.histogram(img_array[:,:,1], bins=32, range=(0, 256))[0]
                b_hist = np.histogram(img_array[:,:,2], bins=32, range=(0, 256))[0]
                
                # Combine and normalize
                features = np.concatenate([r_hist, g_hist, b_hist])
                
                # Pad to 512 dimensions
                if len(features) < 512:
                    features = np.pad(features, (0, 512 - len(features)))
                else:
                    features = features[:512]
                
                # Normalize
                features = features / (np.linalg.norm(features) + 1e-8)
                
                return features.tolist()
        
        except Exception as e:
            print(f"❌ Error creating embedding: {e}")
            traceback.print_exc()
            return None
    
    def store_property_embedding(
        self,
        property_id: str,
        address: str,
        image_path: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        FIXED: Store property embedding with better error handling
        """
        if not self.enabled or not self.client:
            print("⚠️ Supabase not enabled")
            return False
        
        try:
            # Generate embedding
            embedding = self.image_to_embedding(image_path)
            if not embedding:
                print("❌ Failed to generate embedding")
                return False
            
            # Prepare data
            data = {
                "property_id": property_id,
                "address": address,
                "embedding": embedding,
                "metadata": metadata or {}
            }
            
            # Insert or update
            try:
                # Try insert
                response = self.client.table(self.table_name).insert(data).execute()
                print(f"✅ Stored embedding for property {property_id}")
                return True
            
            except Exception as e:
                # If property exists, update instead
                if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                    response = self.client.table(self.table_name).update(data).eq(
                        "property_id", property_id
                    ).execute()
                    print(f"✅ Updated embedding for property {property_id}")
                    return True
                else:
                    raise
        
        except Exception as e:
            print(f"❌ Error storing embedding: {e}")
            traceback.print_exc()
            return False
    
    def find_similar_properties(
        self,
        image_path: str,
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        FIXED: Find similar properties using vector similarity
        """
        if not self.enabled or not self.client:
            print("⚠️ Supabase not enabled")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.image_to_embedding(image_path)
            if not query_embedding:
                print("❌ Failed to generate query embedding")
                return []
            
            # Use Supabase RPC for vector similarity search
            # This requires a custom SQL function - see setup instructions
            try:
                response = self.client.rpc(
                    'match_properties',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': threshold,
                        'match_count': limit
                    }
                ).execute()
                
                return response.data if response.data else []
            
            except Exception as rpc_error:
                print(f"⚠️ RPC function not available: {rpc_error}")
                print("   Using Python-based similarity search (slower)")
                
                # Fallback: Get all embeddings and compute similarity in Python
                response = self.client.table(self.table_name).select("*").execute()
                all_properties = response.data
                
                similar_properties = []
                query_vec = np.array(query_embedding)
                
                for prop in all_properties:
                    if "embedding" in prop and prop["embedding"]:
                        try:
                            prop_vec = np.array(prop["embedding"])
                            
                            # Cosine similarity
                            similarity = np.dot(query_vec, prop_vec) / (
                                np.linalg.norm(query_vec) * np.linalg.norm(prop_vec) + 1e-8
                            )
                            
                            if similarity >= threshold:
                                prop["similarity"] = float(similarity)
                                similar_properties.append(prop)
                        except Exception:
                            continue
                
                # Sort by similarity
                similar_properties.sort(key=lambda x: x["similarity"], reverse=True)
                
                return similar_properties[:limit]
        
        except Exception as e:
            print(f"❌ Error finding similar properties: {e}")
            traceback.print_exc()
            return []
    
    def get_property_by_id(self, property_id: str) -> Optional[Dict]:
        """
        Get property embedding by ID
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            response = self.client.table(self.table_name).select("*").eq(
                "property_id", property_id
            ).execute()
            
            if response.data:
                return response.data[0]
            return None
        
        except Exception as e:
            print(f"❌ Error getting property: {e}")
            return None
    
    def delete_property(self, property_id: str) -> bool:
        """
        Delete property embedding
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            response = self.client.table(self.table_name).delete().eq(
                "property_id", property_id
            ).execute()
            
            print(f"✅ Deleted embedding for property {property_id}")
            return True
        
        except Exception as e:
            print(f"❌ Error deleting property: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics
        """
        if not self.enabled or not self.client:
            return {"error": "Supabase not enabled"}
        
        try:
            # Get total count
            response = self.client.table(self.table_name).select(
                "*", count="exact"
            ).execute()
            
            total_count = response.count if hasattr(response, 'count') else 0
            
            return {
                "total_properties": total_count,
                "table_name": self.table_name,
                "embedding_dimension": 512,
                "enabled": self.enabled
            }
        
        except Exception as e:
            return {"error": str(e)}


# Global instance
vector_db = SupabaseVectorDB()