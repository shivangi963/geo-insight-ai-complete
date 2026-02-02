"""
CLEANUP: Remove ALL /api/properties endpoints and add ONE clean working endpoint
"""
import re

MAIN_PY = "backend/app/main.py"

print("="*80)
print("CLEANUP AND FIX /api/properties ENDPOINT")
print("="*80)

# Read file
with open(MAIN_PY, 'r', encoding='utf-8') as f:
    content = f.read()

print("✅ Read main.py")

# Count current endpoints
count_before = content.count('@app.get("/api/properties")')
print(f"⚠️  Found {count_before} /api/properties endpoints (should be 1)")

# STEP 1: Remove ALL /api/properties endpoints (not /api/properties/raw)
# This pattern matches from @app.get("/api/properties") to the next @app. or major structure
pattern = r'@app\.get\("/api/properties"\)\s*\n.*?(?=@app\.|if __name__|class [A-Z]|^# |^\nclass |^\n@|$)'

# Remove all matches
cleaned_content = re.sub(pattern, '', content, flags=re.DOTALL | re.MULTILINE)

# Verify removal
count_after = cleaned_content.count('@app.get("/api/properties")')
print(f"✅ Removed endpoints. Count after: {count_after}")

# STEP 2: Find insertion point (after /api/properties/raw)
raw_endpoint_pos = cleaned_content.find('@app.get("/api/properties/raw")')

if raw_endpoint_pos == -1:
    print("❌ Could not find /api/properties/raw endpoint!")
    exit(1)

# Find the end of the /raw endpoint (next @app. decorator)
next_decorator = cleaned_content.find('\n@app.', raw_endpoint_pos + 10)

if next_decorator == -1:
    print("❌ Could not find insertion point!")
    exit(1)

# STEP 3: Insert ONE clean working endpoint
CLEAN_ENDPOINT = '''
@app.get("/api/properties")
async def get_properties(skip: int = 0, limit: int = 100):
    """Get all properties with pagination"""
    try:
        logger.info(f"🔍 GET /api/properties called (skip={skip}, limit={limit})")
        
        from backend.app.database import get_database
        db = await get_database()
        
        # Query with pagination
        cursor = db["properties"].find().skip(skip).limit(limit)
        properties = []
        
        async for doc in cursor:
            # Convert _id to id
            if "_id" in doc:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
            properties.append(doc)
        
        logger.info(f"✅ Returned {len(properties)} properties")
        return properties
        
    except Exception as e:
        logger.error(f"Failed to get properties: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

'''

final_content = cleaned_content[:next_decorator] + CLEAN_ENDPOINT + cleaned_content[next_decorator:]

# Write back
with open(MAIN_PY, 'w', encoding='utf-8') as f:
    f.write(final_content)

print("✅ Wrote cleaned file with ONE endpoint")

# Verify
with open(MAIN_PY, 'r', encoding='utf-8') as f:
    verify = f.read()

final_count = verify.count('@app.get("/api/properties")')
has_log = '🔍 GET /api/properties called' in verify

print("\n" + "="*80)
print("VERIFICATION")
print("="*80)
print(f"Endpoint count: {final_count} (should be 1)")
print(f"Has debug log: {has_log}")

if final_count == 1 and has_log:
    print("\n✅✅✅ SUCCESS! File is clean and fixed!")
    print("\nNow:")
    print("1. Start backend: python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000")
    print("2. Should start WITHOUT syntax errors")
    print("3. Test: curl http://localhost:8000/api/properties?limit=5")
    print("4. Refresh Streamlit")
else:
    print("\n❌ Verification failed!")