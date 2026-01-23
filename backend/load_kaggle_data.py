"""
GeoInsight AI - Kaggle Dataset Loader
Load real estate data from CSV into MongoDB
Place this file in: backend/load_kaggle_data.py
"""
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import os
import sys
from geopy.geocoders import Nominatim
import time

class KaggleDataLoader:
    """Load and process Kaggle real estate datasets"""
    
    def __init__(self, mongodb_url="mongodb://localhost:27017", db_name="geoinsight_ai"):
        self.client = MongoClient(mongodb_url, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]
        self.collection = self.db["properties"]
        self.geolocator = Nominatim(user_agent="geoinsight_loader")
        
    def geocode_location(self, address, city, state):
        """Get coordinates for address"""
        try:
            full_address = f"{address}, {city}, {state}"
            location = self.geolocator.geocode(full_address)
            time.sleep(1)  # Rate limiting
            
            if location:
                return location.latitude, location.longitude
            else:
                # Fallback to city-level geocoding
                location = self.geolocator.geocode(f"{city}, {state}")
                time.sleep(1)
                if location:
                    return location.latitude, location.longitude
        except Exception as e:
            print(f"⚠️ Geocoding error: {e}")
        
        return None, None
    
    def clean_price(self, price_str):
        """Clean price string (handles Cr, L, K formats)"""
        if pd.isna(price_str):
            return None
        
        price_str = str(price_str).strip().upper()
        
        try:
            # Remove currency symbols
            price_str = price_str.replace('₹', '').replace('$', '').replace(',', '').strip()
            
            # Handle Crore (Cr)
            if 'CR' in price_str:
                num = float(price_str.replace('CR', '').strip())
                return num * 10000000  # 1 Cr = 10M
            
            # Handle Lakh (L)
            elif 'L' in price_str:
                num = float(price_str.replace('L', '').strip())
                return num * 100000  # 1 L = 100K
            
            # Handle Thousand (K)
            elif 'K' in price_str:
                num = float(price_str.replace('K', '').strip())
                return num * 1000
            
            # Plain number
            else:
                return float(price_str)
        except:
            return None
    
    def load_csv(self, csv_path, clear_existing=False, max_rows=None, geocode=False):
        """Load CSV into MongoDB"""
        
        # Check file exists
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            print("\n📁 Available files in data/:")
            if os.path.exists('data'):
                for f in os.listdir('data'):
                    if f.endswith('.csv'):
                        print(f"   • {f}")
            return False
        
        print(f"\n📂 Reading {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"✅ Found {len(df)} rows")
        print(f"📊 Columns: {', '.join(df.columns.tolist())}")
        
        # Limit rows if specified
        if max_rows:
            df = df.head(max_rows)
            print(f"📌 Limited to {max_rows} rows")
        
        # Clear existing data
        if clear_existing:
            deleted = self.collection.delete_many({})
            print(f"🗑️ Deleted {deleted.deleted_count} existing properties")
        
        # Column mapping (common variations)
        column_map = {
            'address': ['address', 'location', 'property_name', 'name'],
            'city': ['city', 'locality', 'area', 'location'],
            'state': ['state', 'region'],
            'price': ['price', 'cost', 'value', 'amount'],
            'bedrooms': ['bedrooms', 'bhk', 'beds', 'bedroom'],
            'bathrooms': ['bathrooms', 'baths', 'bathroom'],
            'square_feet': ['square_feet', 'area', 'sqft', 'size', 'carpet_area'],
            'property_type': ['property_type', 'type', 'category']
        }
        
        def find_column(variations):
            """Find first matching column"""
            for var in variations:
                for col in df.columns:
                    if var.lower() in col.lower():
                        return col
            return None
        
        # Map columns
        mapped = {}
        for key, variations in column_map.items():
            col = find_column(variations)
            if col:
                mapped[key] = col
                print(f"✓ Mapped '{key}' → '{col}'")
        
        if not mapped.get('city'):
            print("❌ Could not find 'city' column. Please check your CSV.")
            return False
        
        # Process data
        properties_added = 0
        errors = 0
        
        print(f"\n⚙️ Processing properties...")
        
        for idx, row in df.iterrows():
            try:
                # Extract data with fallbacks
                city = str(row.get(mapped.get('city', 'city'), 'Unknown'))
                state = str(row.get(mapped.get('state', 'state'), 'Unknown'))
                
                # Handle address
                address_col = mapped.get('address')
                if address_col and address_col in row:
                    address = str(row[address_col])
                else:
                    address = f"Property in {city}"
                
                # Handle price
                price_col = mapped.get('price')
                if price_col and price_col in row:
                    price = self.clean_price(row[price_col])
                else:
                    price = None
                
                if price is None or price <= 0:
                    price = 300000  # Default fallback
                
                # Handle bedrooms
                bed_col = mapped.get('bedrooms')
                if bed_col and bed_col in row:
                    bedrooms = int(float(str(row[bed_col]).replace('BHK', '').strip()[0]))
                else:
                    bedrooms = 2
                
                # Handle bathrooms
                bath_col = mapped.get('bathrooms')
                bathrooms = float(row.get(bath_col, 2.0)) if bath_col else 2.0
                
                # Handle square feet
                sqft_col = mapped.get('square_feet')
                if sqft_col and sqft_col in row:
                    square_feet = int(float(row[sqft_col]))
                else:
                    square_feet = int(price / 250)  # Estimate
                
                # Property type
                type_col = mapped.get('property_type')
                property_type = str(row.get(type_col, 'Apartment')) if type_col else 'Apartment'
                
                # Geocoding (optional, slow)
                if geocode and properties_added % 10 == 0:  # Only geocode every 10th property
                    lat, lon = self.geocode_location(address, city, state)
                else:
                    lat, lon = None, None
                
                # Default coordinates if not geocoded
                if lat is None:
                    # India defaults (can be adjusted)
                    lat = 19.0760  # Mumbai
                    lon = 72.8777
                
                # Create document
                property_doc = {
                    "address": address,
                    "city": city,
                    "state": state,
                    "zip_code": "000000",
                    "price": float(price),
                    "bedrooms": int(bedrooms),
                    "bathrooms": float(bathrooms),
                    "square_feet": int(square_feet),
                    "property_type": property_type,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                
                # Insert
                self.collection.insert_one(property_doc)
                properties_added += 1
                
                # Progress
                if properties_added % 50 == 0:
                    print(f"✅ Loaded {properties_added} properties...")
                
            except Exception as e:
                errors += 1
                if errors <= 5:  # Only show first 5 errors
                    print(f"⚠️ Error on row {idx}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"✅ Successfully loaded {properties_added} properties!")
        print(f"⚠️ Errors: {errors}")
        print(f"📊 Total in database: {self.collection.count_documents({})}")
        print(f"{'='*60}")
        
        # Show sample
        self.show_sample()
        
        return True
    
    def show_sample(self, limit=5):
        """Show sample properties"""
        print(f"\n📋 Sample properties:")
        for prop in self.collection.find().limit(limit):
            print(f"  • {prop['address']}, {prop['city']}, {prop['state']}")
            print(f"    ${prop['price']:,.0f} | {prop['bedrooms']}bed | {prop['square_feet']}sqft")
        
        # City breakdown
        print(f"\n🌆 Cities loaded:")
        pipeline = [
            {"$group": {"_id": "$city", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        for doc in self.collection.aggregate(pipeline):
            print(f"  • {doc['_id']}: {doc['count']} properties")
    
    def close(self):
        """Close connection"""
        self.client.close()
        print("\n✅ Connection closed")


def main():
    """Main execution"""
    print("="*60)
    print("  GeoInsight AI - Kaggle Data Loader")
    print("="*60)
    
    # Initialize loader
    loader = KaggleDataLoader()
    
    # Interactive mode
    print("\n📁 Available CSV files:")
    csv_files = []
    if os.path.exists('data'):
        for f in os.listdir('data'):
            if f.endswith('.csv'):
                csv_files.append(f)
                print(f"  {len(csv_files)}. data/{f}")
    
    if not csv_files:
        print("\n❌ No CSV files found in data/ folder")
        print("\nPlease add your Kaggle dataset to:")
        print("  data/your_dataset.csv")
        return
    
    # Get user input
    print("\n" + "="*60)
    choice = input(f"Select file (1-{len(csv_files)}) or enter path: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
        csv_path = os.path.join('data', csv_files[int(choice)-1])
    else:
        csv_path = choice
    
    # Options
    print("\n" + "="*60)
    clear = input("Clear existing properties? (y/n): ").lower() == 'y'
    
    max_rows_input = input("Max rows to load (press Enter for all): ").strip()
    max_rows = int(max_rows_input) if max_rows_input else None
    
    geocode = input("Geocode addresses? (slow, y/n): ").lower() == 'y'
    
    # Load data
    print("\n" + "="*60)
    success = loader.load_csv(
        csv_path=csv_path,
        clear_existing=clear,
        max_rows=max_rows,
        geocode=geocode
    )
    
    if success:
        print("\n✅ Data loaded successfully!")
        print("\n🚀 Next steps:")
        print("  1. Restart your backend: uvicorn app.main:app --reload")
        print("  2. Refresh Streamlit dashboard")
        print("  3. You should now see all loaded properties!")
    
    loader.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()