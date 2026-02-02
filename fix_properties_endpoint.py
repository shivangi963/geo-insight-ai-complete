"""
Fix /api/properties endpoint to accept GET requests
"""
import re

MAIN_PY = "backend/app/main.py"

# The working endpoint code
WORKING_ENDPOINT = '''
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

print("="*80)
print("FIXING /api/properties ENDPOINT")
print("="*80)

# Read file
with open(MAIN_PY, 'r', encoding='utf-8') as f:
    content = f.read()

print("✅ Read main.py")

# Check if the endpoint already exists as GET
if '@app.get("/api/properties")' in content:
    print("⚠️  Endpoint already exists as GET, checking if it works...")
    # Find and show it
    pattern = r'@app\.get\("/api/properties"\).*?(?=@app\.|if __name__|class |def [a-z_]+\(|$)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        existing = match.group(0)
        print(f"Existing endpoint ({len(existing)} chars):")
        print(existing[:500])
        print("\nReplacing with working version...")
        new_content = content.replace(existing, WORKING_ENDPOINT)
    else:
        print("❌ Could not find existing endpoint to replace")
        new_content = content
else:
    # Find where to insert it (after /api/properties/raw)
    insertion_point = content.find('@app.get("/api/properties/raw")')
    
    if insertion_point != -1:
        # Find the end of the /raw endpoint
        next_endpoint = content.find('@app.', insertion_point + 10)
        if next_endpoint != -1:
            # Insert before next endpoint
            new_content = content[:next_endpoint] + WORKING_ENDPOINT + '\n\n' + content[next_endpoint:]
            print("✅ Inserted new GET /api/properties endpoint")
        else:
            print("❌ Could not find insertion point")
            new_content = content
    else:
        print("❌ Could not find /api/properties/raw to insert after")
        new_content = content

# Write back
with open(MAIN_PY, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Wrote changes to main.py")

# Verify
with open(MAIN_PY, 'r', encoding='utf-8') as f:
    verify_content = f.read()

if '@app.get("/api/properties")' in verify_content and 'GET /api/properties called' in verify_content:
    print("✅ VERIFIED: GET /api/properties endpoint is now in the file!")
    print("\n" + "="*80)
    print("SUCCESS! Endpoint fixed.")
    print("="*80)
    print("\nNow:")
    print("1. Restart backend (Ctrl+C in backend window)")
    print("2. Start backend again")
    print("3. Refresh Streamlit page")
    print("4. Properties should load!")
else:
    print("❌ Verification failed")