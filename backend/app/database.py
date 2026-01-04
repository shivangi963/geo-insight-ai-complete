import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

class Database:
    
    client= None
    db = None
    
    @classmethod
    async def connect(cls):
     
        if cls.client is None:
            mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            database_name = os.getenv("DATABASE_NAME", "geoinsight_ai")
            
            cls.client = AsyncIOMotorClient(mongodb_url)
            cls.db = cls.client[database_name]
            print(" Connected to MongoDB asynchronously!")
        
        return cls.db
    
    @classmethod
    async def get_database(cls):
    
        if cls.db is None:
            await cls.connect()
        return cls.db
    
    @classmethod
    async def get_collection(cls, collection_name: str):
     
        db = await cls.get_database()
        return db[collection_name]
    
    @classmethod
    async def close(cls):
     
        if cls.client:
            cls.client.close()
            print("Disconnected from MongoDB")

async def get_database():
 
    return await Database.get_database()


def get_sync_database():

    from pymongo import MongoClient
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name = os.getenv("DATABASE_NAME", "geoinsight_ai")
    client = MongoClient(mongodb_url)
    return client[database_name]
