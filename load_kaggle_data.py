import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import os

def load_kaggle_dataset():
    
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["geoinsight_ai"]
    collection = db["properties"]
    
    # Path to your cleaned Kaggle dataset
    csv_path = "data/Mumbai House Price.csv" 
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    
    df = pd.read_csv(csv_path)
    
    print(f"✅ Found {len(df)} properties")
    
    properties_added = 0
    
    for idx, row in df.iterrows():
        try:
            # Create property document
            property_doc = {
                "address": str(row.get('address', f'Property {idx}')),
                "city": str(row.get('city', row.get('location', 'Mumbai'))),
                "state": str(row.get('state', 'Maharashtra')),
                "zip_code": str(row.get('zip_code', '400001')),
                "price": float(row.get('price', 0)),
                "bedrooms": int(row.get('bedrooms', row.get('bhk', 2))),
                "bathrooms": float(row.get('bathrooms', 2.0)),
                "square_feet": int(row.get('square_feet', row.get('area', 1000))),
                "property_type": str(row.get('property_type', 'Apartment')),
                "latitude": float(row.get('latitude', 19.0760)),  # Mumbai default
                "longitude": float(row.get('longitude', 72.8777)),
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            # Insert into MongoDB
            collection.insert_one(property_doc)
            properties_added += 1
            
            if properties_added % 100 == 0:
                print(f"✅ Loaded {properties_added} properties...")
                
        except Exception as e:
            print(f"⚠️ Error loading row {idx}: {e}")
            continue
    
    print(f"\n✅ Successfully loaded {properties_added} properties!")
    print(f"📊 Total properties in database: {collection.count_documents({})}")
    
    # Show sample
    print("\n📋 Sample properties:")
    for prop in collection.find().limit(3):
        print(f"  • {prop['address']}, {prop['city']} - ${prop['price']:,.0f}")
    
    client.close()

if __name__ == "__main__":
    print("=" * 50)
    print("GeoInsight AI - Kaggle Data Loader")
    print("=" * 50)
    load_kaggle_dataset()