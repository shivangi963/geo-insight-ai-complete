from database import Database
from models import PropertyCreate
from datetime import datetime

def load_sample_data():
    collection = Database.get_collection("properties")
    

    collection.delete_many({})
    
    sample_properties = [
        PropertyCreate(
            address="123 Main St",
            city="San Francisco",
            state="CA",
            zip_code="94105",
            price=1500000,
            bedrooms=3,
            bathrooms=2.5,
            square_feet=1800,
            property_type="Single Family",
            latitude=37.7749,
            longitude=-122.4194
        ),
        PropertyCreate(
            address="456 Oak Ave",
            city="New York",
            state="NY",
            zip_code="10001",
            price=1200000,
            bedrooms=2,
            bathrooms=2.0,
            square_feet=1200,
            property_type="Condo",
            latitude=40.7128,
            longitude=-74.0060
        ),
        PropertyCreate(
            address="789 Pine Rd",
            city="Austin",
            state="TX",
            zip_code="73301",
            price=750000,
            bedrooms=4,
            bathrooms=3.0,
            square_feet=2200,
            property_type="Single Family",
            latitude=30.2672,
            longitude=-97.7431
        )
    ]
    

    for property in sample_properties:
        property_dict = property.model_dump()
        property_dict["created_at"] = datetime.utcnow()
        property_dict["updated_at"] = datetime.utcnow()
        collection.insert_one(property_dict)
    
    print(" Sample data loaded successfully!")
    count = collection.count_documents({})
    print(f"Total properties in database: {count}")

if __name__ == "__main__":
    load_sample_data()