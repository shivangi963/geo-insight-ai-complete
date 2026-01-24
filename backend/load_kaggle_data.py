"""
FIXED: Kaggle Data Loader - Scriptable and Async-friendly
Now supports command-line arguments and batch processing
"""
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import os
import sys
import argparse
from typing import Optional, Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Optional geocoding
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOCODING_AVAILABLE = True
except ImportError:
    GEOCODING_AVAILABLE = False
    print("⚠️ Geopy not installed. Geocoding disabled.")


class KaggleDataLoader:
    """
    FIXED: Load and process Kaggle real estate datasets
    Now supports batch processing and CLI arguments
    """
    
    def __init__(
        self,
        mongodb_url: str = "mongodb://localhost:27017",
        db_name: str = "geoinsight_ai"
    ):
        self.client = MongoClient(mongodb_url, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]
        self.collection = self.db["properties"]
        self.geolocator = None
        
        if GEOCODING_AVAILABLE:
            self.geolocator = Nominatim(
                user_agent="geoinsight_loader",
                timeout=10
            )
    
    def geocode_location(
        self,
        address: str,
        city: str,
        state: str,
        retry_count: int = 2
    ) -> tuple:
        """
        FIXED: Geocode with retry logic and better error handling
        """
        if not self.geolocator:
            return None, None
        
        for attempt in range(retry_count):
            try:
                full_address = f"{address}, {city}, {state}"
                location = self.geolocator.geocode(full_address)
                
                if location:
                    return location.latitude, location.longitude
                
                # Fallback to city-level
                location = self.geolocator.geocode(f"{city}, {state}")
                if location:
                    return location.latitude, location.longitude
                
                return None, None
            
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                if attempt < retry_count - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                print(f"⚠️ Geocoding failed: {e}")
                return None, None
            
            except Exception as e:
                print(f"⚠️ Geocoding error: {e}")
                return None, None
        
        return None, None
    
    def clean_price(self, price_str) -> Optional[float]:
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
    
    def detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect column names from CSV
        """
        column_map = {
            'address': ['address', 'location', 'property_name', 'name', 'street'],
            'city': ['city', 'locality', 'area', 'town'],
            'state': ['state', 'region', 'province'],
            'price': ['price', 'cost', 'value', 'amount', 'rate'],
            'bedrooms': ['bedrooms', 'bhk', 'beds', 'bedroom', 'bed'],
            'bathrooms': ['bathrooms', 'baths', 'bathroom', 'bath'],
            'square_feet': ['square_feet', 'area', 'sqft', 'size', 'carpet_area', 'built_up_area'],
            'property_type': ['property_type', 'type', 'category', 'kind']
        }
        
        def find_column(variations: List[str]) -> Optional[str]:
            for var in variations:
                for col in df.columns:
                    if var.lower() in col.lower():
                        return col
            return None
        
        mapped = {}
        for key, variations in column_map.items():
            col = find_column(variations)
            if col:
                mapped[key] = col
        
        return mapped
    
    def load_csv(
        self,
        csv_path: str,
        clear_existing: bool = False,
        max_rows: Optional[int] = None,
        geocode: bool = False,
        geocode_batch_size: int = 10,
        verbose: bool = True
    ) -> bool:
        """
        FIXED: Load CSV with batch processing and better control
        """
        # Validate file
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            return False
        
        if verbose:
            print(f"\n📂 Reading {csv_path}...")
        
        df = pd.read_csv(csv_path)
        
        if verbose:
            print(f"✅ Found {len(df)} rows")
            print(f"📊 Columns: {', '.join(df.columns.tolist())}")
        
        # Limit rows
        if max_rows:
            df = df.head(max_rows)
            if verbose:
                print(f"📌 Limited to {max_rows} rows")
        
        # Clear existing
        if clear_existing:
            deleted = self.collection.delete_many({})
            if verbose:
                print(f"🗑️ Deleted {deleted.deleted_count} existing properties")
        
        # Auto-detect columns
        mapped = self.detect_columns(df)
        
        if not mapped.get('city'):
            print("❌ Could not find 'city' column")
            return False
        
        if verbose:
            print("\n📋 Column Mapping:")
            for key, col in mapped.items():
                print(f"  • {key} → {col}")
        
        # Process data
        properties_added = 0
        errors = 0
        
        if verbose:
            print(f"\n⚙️ Processing properties...")
        
        # Default coordinates (can be overridden)
        default_coords = {
            'India': (20.5937, 78.9629),
            'US': (37.0902, -95.7129),
            'UK': (55.3781, -3.4360)
        }
        
        for idx, row in df.iterrows():
            try:
                # Extract data
                city = str(row.get(mapped.get('city', 'city'), 'Unknown'))
                state = str(row.get(mapped.get('state', 'state'), 'Unknown'))
                
                # Address
                address_col = mapped.get('address')
                if address_col and address_col in row:
                    address = str(row[address_col])
                else:
                    address = f"Property in {city}"
                
                # Price
                price_col = mapped.get('price')
                if price_col and price_col in row:
                    price = self.clean_price(row[price_col])
                else:
                    price = None
                
                if price is None or price <= 0:
                    price = 300000  # Default
                
                # Bedrooms
                bed_col = mapped.get('bedrooms')
                if bed_col and bed_col in row:
                    try:
                        bed_str = str(row[bed_col]).replace('BHK', '').strip()
                        bedrooms = int(float(bed_str[0]))
                    except:
                        bedrooms = 2
                else:
                    bedrooms = 2
                
                # Bathrooms
                bath_col = mapped.get('bathrooms')
                try:
                    bathrooms = float(row.get(bath_col, 2.0)) if bath_col else 2.0
                except:
                    bathrooms = 2.0
                
                # Square feet
                sqft_col = mapped.get('square_feet')
                if sqft_col and sqft_col in row:
                    try:
                        square_feet = int(float(row[sqft_col]))
                    except:
                        square_feet = int(price / 250)
                else:
                    square_feet = int(price / 250)
                
                # Property type
                type_col = mapped.get('property_type')
                property_type = str(row.get(type_col, 'Apartment')) if type_col else 'Apartment'
                
                # Geocoding (batch-based)
                if geocode and properties_added % geocode_batch_size == 0:
                    lat, lon = self.geocode_location(address, city, state)
                    if lat is None:
                        # Use default for country
                        lat, lon = default_coords.get('India')
                else:
                    lat, lon = default_coords.get('India')
                
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
                if verbose and properties_added % 50 == 0:
                    print(f"✅ Loaded {properties_added} properties...")
            
            except Exception as e:
                errors += 1
                if verbose and errors <= 5:
                    print(f"⚠️ Error on row {idx}: {e}")
                continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ Successfully loaded {properties_added} properties!")
            print(f"⚠️ Errors: {errors}")
            print(f"📊 Total in database: {self.collection.count_documents({})}")
            print(f"{'='*60}")
            
            self.show_sample()
        
        return True
    
    def show_sample(self, limit: int = 5):
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


def main():
    """
    FIXED: CLI interface with argparse
    """
    parser = argparse.ArgumentParser(
        description="Load Kaggle real estate data into MongoDB"
    )
    
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to CSV file'
    )
    
    parser.add_argument(
        '--mongodb-url',
        type=str,
        default='mongodb://localhost:27017',
        help='MongoDB connection URL'
    )
    
    parser.add_argument(
        '--db-name',
        type=str,
        default='geoinsight_ai',
        help='Database name'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing properties before loading'
    )
    
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum rows to load'
    )
    
    parser.add_argument(
        '--geocode',
        action='store_true',
        help='Enable geocoding (slow)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Geocoding batch size'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Initialize loader
    print("="*60)
    print("  GeoInsight AI - Kaggle Data Loader")
    print("="*60)
    
    loader = KaggleDataLoader(
        mongodb_url=args.mongodb_url,
        db_name=args.db_name
    )
    
    # Load data
    success = loader.load_csv(
        csv_path=args.csv_path,
        clear_existing=args.clear,
        max_rows=args.max_rows,
        geocode=args.geocode,
        geocode_batch_size=args.batch_size,
        verbose=not args.quiet
    )
    
    if success:
        print("\n✅ Data loaded successfully!")
        print("\n🚀 Next steps:")
        print("  1. Restart backend: uvicorn app.main:app --reload")
        print("  2. Refresh Streamlit dashboard")
        print("  3. Properties are now available!")
    
    loader.close()


if __name__ == "__main__":
    try:
        # If run without arguments, use interactive mode
        if len(sys.argv) == 1:
            print("="*60)
            print("  GeoInsight AI - Kaggle Data Loader")
            print("="*60)
            
            loader = KaggleDataLoader()
            
            # Find CSV files
            csv_files = []
            if os.path.exists('data'):
                for f in os.listdir('data'):
                    if f.endswith('.csv'):
                        csv_files.append(f)
                        print(f"  {len(csv_files)}. data/{f}")
            
            if not csv_files:
                print("\n❌ No CSV files found in data/ folder")
                print("\nUsage: python load_kaggle_data.py <csv_path> [options]")
                print("Run: python load_kaggle_data.py --help for options")
                sys.exit(1)
            
            # Interactive selection
            choice = input(f"\nSelect file (1-{len(csv_files)}): ").strip()
            csv_path = os.path.join('data', csv_files[int(choice)-1])
            
            clear = input("Clear existing? (y/n): ").lower() == 'y'
            max_rows = input("Max rows (Enter for all): ").strip()
            max_rows = int(max_rows) if max_rows else None
            
            success = loader.load_csv(
                csv_path=csv_path,
                clear_existing=clear,
                max_rows=max_rows
            )
            
            loader.close()
        else:
            # Use CLI mode
            main()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()