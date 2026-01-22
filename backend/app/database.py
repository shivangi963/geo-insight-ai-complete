import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from dotenv import load_dotenv
import asyncio

load_dotenv()

class Database:
    """MongoDB Database Manager - FIXED VERSION"""
    client = None
    db = None
    _is_connected = False
    
    @classmethod
    async def connect(cls):
        """Connect to MongoDB asynchronously"""
        if cls.client is None:
            mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            database_name = os.getenv("DATABASE_NAME", "geoinsight_ai")
            
            try:
                cls.client = AsyncIOMotorClient(mongodb_url, serverSelectionTimeoutMS=5000)
                cls.db = cls.client[database_name]
                
                # Test connection
                await cls.client.admin.command('ping')
                cls._is_connected = True
                print(f"✅ Connected to MongoDB successfully!")
                print(f"   Database: {database_name}")
                print(f"   URL: {mongodb_url}")
                
            except Exception as e:
                cls._is_connected = False
                print(f"❌ MongoDB connection failed: {e}")
                print("   Make sure MongoDB is running: 'net start MongoDB'")
                raise
        
        return cls.db
    
    @classmethod
    async def is_connected(cls) -> bool:
        """Check if database is connected - IMPROVED"""
        if not cls._is_connected or cls.db is None:
            return False
        
        try:
            # Actually ping the database to verify connection
            await cls.client.admin.command('ping')
            return True
        except Exception as e:
            print(f"⚠️ Database connection check failed: {e}")
            cls._is_connected = False
            return False
    
    @classmethod
    async def get_database(cls):
        """Get database instance"""
        if cls.db is None or not cls._is_connected:
            await cls.connect()
        return cls.db
    
    @classmethod
    async def get_collection(cls, collection_name: str):
        """Get a specific collection"""
        db = await cls.get_database()
        return db[collection_name]
    
    @classmethod
    async def close(cls):
        """Close database connection"""
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
            cls._is_connected = False
            print("✅ Disconnected from MongoDB")

async def get_database():
    """Helper function to get database"""
    return await Database.get_database()

def get_sync_database():
    """Get synchronous database connection (for Celery tasks)"""
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name = os.getenv("DATABASE_NAME", "geoinsight_ai")
    
    try:
        client = MongoClient(mongodb_url, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        print(f"✅ Sync connection to MongoDB successful!")
        return client[database_name]
    except Exception as e:
        print(f"❌ Sync MongoDB connection failed: {e}")
        raise

# Initialize collections on startup
async def initialize_database():
    """Initialize database with required collections and indexes"""
    try:
        db = await get_database()
        
        # Create collections if they don't exist
        collections = await db.list_collection_names()
        
        if "properties" not in collections:
            await db.create_collection("properties")
            print("✅ Created 'properties' collection")
        
        if "neighborhood_analyses" not in collections:
            await db.create_collection("neighborhood_analyses")
            print("✅ Created 'neighborhood_analyses' collection")
        
        # Create indexes for better performance
        await db.properties.create_index("address")
        await db.properties.create_index("city")
        await db.properties.create_index([("latitude", 1), ("longitude", 1)])
        
        await db.neighborhood_analyses.create_index("created_at")
        await db.neighborhood_analyses.create_index("status")
        await db.neighborhood_analyses.create_index("address")
        
        print("✅ Database indexes created")
        
        # Load sample data if properties collection is empty
        count = await db.properties.count_documents({})
        if count == 0:
            await load_sample_properties(db)
        
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

async def load_sample_properties(db):
    """Load sample property data"""
    from datetime import datetime
    
    sample_properties = [
        {
            "address": "123 Main St",
            "city": "San Francisco",
            "state": "CA",
            "zip_code": "94105",
            "price": 1500000,
            "bedrooms": 3,
            "bathrooms": 2.5,
            "square_feet": 1800,
            "property_type": "Single Family",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "address": "456 Oak Ave",
            "city": "New York",
            "state": "NY",
            "zip_code": "10001",
            "price": 1200000,
            "bedrooms": 2,
            "bathrooms": 2.0,
            "square_feet": 1200,
            "property_type": "Condo",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "address": "789 Pine Rd",
            "city": "Austin",
            "state": "TX",
            "zip_code": "73301",
            "price": 750000,
            "bedrooms": 4,
            "bathrooms": 3.0,
            "square_feet": 2200,
            "property_type": "Single Family",
            "latitude": 30.2672,
            "longitude": -97.7431,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        },
        {
            "address": "MIT Campus, Manipal",
            "city": "Manipal",
            "state": "Karnataka",
            "zip_code": "576104",
            "price": 450000,
            "bedrooms": 2,
            "bathrooms": 2.0,
            "square_feet": 1100,
            "property_type": "Apartment",
            "latitude": 13.3519,
            "longitude": 74.7870,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
    ]
    
    result = await db.properties.insert_many(sample_properties)
    print(f"✅ Loaded {len(result.inserted_ids)} sample properties")