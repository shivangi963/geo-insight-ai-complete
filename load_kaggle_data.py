"""
Enhanced Mumbai Housing Data Loader
Loads Mumbai Housing Price dataset into MongoDB with proper mapping
"""
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import os
import sys
import argparse
from typing import Optional, Dict, List
import time

class MumbaiHousingLoader:
    """Load Mumbai Housing dataset into MongoDB"""
    
    def __init__(
        self,
        mongodb_url: str = "mongodb://localhost:27017",
        db_name: str = "geoinsight_ai"
    ):
        self.client = MongoClient(mongodb_url, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]
        self.collection = self.db["properties"]
        
        # No coordinates needed - simplified version
    
    def clean_price_with_unit(self, price_value, price_unit) -> Optional[float]:
        """Clean price using separate price and price_unit columns"""
        if pd.isna(price_value):
            return None
        
        try:
            # Get numeric price
            price = float(price_value)
            
            # Get unit
            if pd.isna(price_unit):
                # If no unit, assume it's already in INR
                return price
            
            unit = str(price_unit).strip().upper()
            
            # Convert based on unit
            if unit == 'CR' or unit == 'CRORE':
                return price * 10000000  # 1 Cr = 10M INR
            elif unit == 'L' or unit == 'LAC' or unit == 'LAKH':
                return price * 100000  # 1 L = 100K INR
            
            elif unit == 'K' or unit == 'THOUSAND':
                return price * 1000
            else:
                # Assume already in INR
                return price
        except:
            return None
    
    def extract_bedrooms(self, bhk_value) -> int:
        """Extract bedroom count from bhk column"""
        if pd.isna(bhk_value):
            return 2
        
        try:
            # Direct integer conversion
            return int(bhk_value)
        except:
            return 2
    
    # Coordinates removed - not needed
    
    def load_mumbai_housing(
        self,
        csv_path: str = "backend/data/Mumbai House Prices.csv",
        clear_existing: bool = True,
        max_rows: Optional[int] = None,
        verbose: bool = True
    ) -> bool:
        """Load Mumbai Housing dataset"""
        
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            return False
        
        if verbose:
            print(f"\n📂 Reading {csv_path}...")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return False
        
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
        
        # Process data
        properties_added = 0
        errors = 0
        
        if verbose:
            print(f"\n⚙️ Processing Mumbai housing data...")
        
        for idx, row in df.iterrows():
            try:
                # Extract data using exact column names
                locality = row.get('locality', 'Mumbai')
                property_type = row.get('type', 'Apartment')
                bhk = row.get('bhk', 2)
                area_sqft = row.get('area', 0)
                price_value = row.get('price', 0)
                price_unit = row.get('price_unit', 'L')
                price_inr = row.get('price_inr', 0)
                region = row.get('region', locality)
                status = row.get('status', 'Ready to move')
                age = row.get('age', 'New')
                price_per_sqft_orig = row.get('price_per_sqft', 0)
                
                city = 'Mumbai'
                state = 'Maharashtra'
                
                # Use price_inr if available, otherwise calculate
                if pd.notna(price_inr) and price_inr > 0:
                    price = float(price_inr)
                else:
                    price = self.clean_price_with_unit(price_value, price_unit)
                
                if price is None or price <= 0:
                    price = 5000000  # Default 50L
                
                # Bedrooms
                bedrooms = self.extract_bedrooms(bhk)
                
                # Bathrooms (approximate)
                bathrooms = max(1.0, bedrooms * 0.75)
                
                # Square feet
                try:
                    square_feet = int(float(area_sqft)) if pd.notna(area_sqft) and area_sqft > 0 else int(price / 15000)
                except:
                    square_feet = int(price / 15000)
                
                # Calculate price per sqft if not provided
                if square_feet > 0:
                    price_per_sqft = price / square_feet
                else:
                    price_per_sqft = price_per_sqft_orig if pd.notna(price_per_sqft_orig) else 0
                
                # Create full address (no coordinates needed)
                address_parts = []
                if pd.notna(locality) and str(locality).strip():
                    address_parts.append(str(locality))
                if pd.notna(region) and str(region).strip() and region != locality:
                    address_parts.append(str(region))
                address = ', '.join(address_parts) + ', Mumbai' if address_parts else 'Mumbai'
                
                # Create document (no latitude/longitude)
                property_doc = {
                    "address": address,
                    "city": city,
                    "state": state,
                    "zip_code": "400001",
                    "price": float(price),
                    "bedrooms": int(bedrooms),
                    "bathrooms": float(bathrooms),
                    "square_feet": int(square_feet),
                    "property_type": str(property_type),
                    "locality": str(locality),
                    "region": str(region) if pd.notna(region) else str(locality),
                    "status": str(status) if pd.notna(status) else "Ready to move",
                    "age": str(age) if pd.notna(age) else "New",
                    "price_per_sqft": float(price_per_sqft),
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                
                # Insert
                self.collection.insert_one(property_doc)
                properties_added += 1
                
                # Progress
                if verbose and properties_added % 100 == 0:
                    print(f"✅ Loaded {properties_added} properties...")
            
            except Exception as e:
                errors += 1
                if verbose and errors <= 5:
                    print(f"⚠️ Error on row {idx}: {e}")
                continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ Successfully loaded {properties_added} Mumbai properties!")
            print(f"⚠️ Errors: {errors}")
            print(f"📊 Total in database: {self.collection.count_documents({})}")
            print(f"{'='*60}")
            
            self.show_stats()
        
        return True
    
    def show_stats(self):
        """Show database statistics"""
        print(f"\n📊 Database Statistics:")
        
        # Total properties
        total = self.collection.count_documents({})
        print(f"  Total properties: {total}")
        
        # By locality (top 10)
        print(f"\n🏘️ Top 10 Localities:")
        pipeline = [
            {"$group": {"_id": "$locality", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        for doc in self.collection.aggregate(pipeline):
            print(f"  • {doc['_id']}: {doc['count']} properties")
        
        # Price statistics
        print(f"\n💰 Price Statistics:")
        pipeline = [
            {"$group": {
                "_id": None,
                "avg_price": {"$avg": "$price"},
                "min_price": {"$min": "$price"},
                "max_price": {"$max": "$price"}
            }}
        ]
        stats = list(self.collection.aggregate(pipeline))
        if stats:
            stat = stats[0]
            print(f"  Average: ₹{stat['avg_price']:,.0f}")
            print(f"  Minimum: ₹{stat['min_price']:,.0f}")
            print(f"  Maximum: ₹{stat['max_price']:,.0f}")
        
        # By BHK
        print(f"\n🛏️ By Bedroom Count:")
        pipeline = [
            {"$group": {"_id": "$bedrooms", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        for doc in self.collection.aggregate(pipeline):
            print(f"  • {doc['_id']} BHK: {doc['count']} properties")
    
    def close(self):
        """Close connection"""
        self.client.close()


def main():
    """Main function with CLI"""
    parser = argparse.ArgumentParser(
        description="Load Mumbai Housing dataset into MongoDB"
    )
    
    parser.add_argument(
        '--csv-path',
        type=str,
        default='backend/data/Mumbai House Prices.csv',
        help='Path to Mumbai Housing CSV file'
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
        '--keep-existing',
        action='store_true',
        help='Keep existing properties (do not clear)'
    )
    
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum rows to load'
    )
    
    args = parser.parse_args()
    
    # Initialize loader
    print("="*60)
    print("  Mumbai Housing Data Loader")
    print("="*60)
    
    loader = MumbaiHousingLoader(
        mongodb_url=args.mongodb_url,
        db_name=args.db_name
    )
    
    # Load data
    success = loader.load_mumbai_housing(
        csv_path=args.csv_path,
        clear_existing=not args.keep_existing,
        max_rows=args.max_rows,
        verbose=True
    )
    
    if success:
        print("\n✅ Data loaded successfully!")
        print("\n🚀 Next steps:")
        print("  1. Restart backend: uvicorn app.main:app --reload")
        print("  2. Refresh Streamlit dashboard")
        print("  3. Browse properties by Mumbai locality!")
    else:
        print("\n❌ Data loading failed!")
    
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