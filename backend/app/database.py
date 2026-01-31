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
            
            print(f"\n🔌 Attempting MongoDB connection...")
            print(f"   URL: {mongodb_url}")
            print(f"   Database: {database_name}")
            
            try:
                cls.client = AsyncIOMotorClient(
                    mongodb_url, 
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000
                )
                cls.db = cls.client[database_name]
                
                # Test connection with retry
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        await cls.client.admin.command('ping')
                        cls._is_connected = True
                        print(f"✅ Connected to MongoDB successfully!")
                        print(f"   Database: {database_name}")
                        
                        # Verify collections exist
                        collections = await cls.db.list_collection_names()
                        print(f"   Collections: {collections}")
                        
                        # Count properties
                        if 'properties' in collections:
                            count = await cls.db.properties.count_documents({})
                            print(f"   Properties count: {count}")
                        
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"   Retry {attempt + 1}/{max_retries}...")
                            await asyncio.sleep(1)
                        else:
                            raise
                
            except Exception as e:
                cls._is_connected = False
                print(f"❌ MongoDB connection failed: {e}")
                print("   Make sure MongoDB is running: 'net start MongoDB'")
                print(f"   Connection string: {mongodb_url}")
                raise
        
        return cls.db
    
    @classmethod
    async def is_connected(cls) -> bool:
        """Check if database is connected"""
        if not cls._is_connected or cls.client is None or cls.db is None:
            return False
        try:
            await cls.client.admin.command("ping")
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
    
    print(f"\n🔌 Sync connection to MongoDB...")
    print(f"   URL: {mongodb_url}")
    print(f"   Database: {database_name}")
    
    try:
        client = MongoClient(
            mongodb_url, 
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000
        )
        # Test connection
        client.admin.command('ping')
        
        db = client[database_name]
        
        # Verify data
        collections = db.list_collection_names()
        print(f"   Collections: {collections}")
        
        if 'properties' in collections:
            count = db.properties.count_documents({})
            print(f"   Properties: {count}")
        
        print(f"✅ Sync connection successful!")
        return db
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
        
        # Check if properties exist
        count = await db.properties.count_documents({})
        print(f"✅ Found {count} properties in database")
        
        if count == 0:
            print("⚠️  WARNING: No properties in database!")
            print("   Run: python load_kaggle_data.py")
        
    except Exception as e:
        print(f" Database initialization error: {e}")
        import traceback
        traceback.print_exc()