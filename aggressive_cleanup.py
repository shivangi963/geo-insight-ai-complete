"""
AGGRESSIVE CLEANUP - Remove ALL broken /api/properties endpoints
"""
import re

MAIN_PY = "backend/app/main.py"

print("="*80)
print("AGGRESSIVE CLEANUP")
print("="*80)

with open(MAIN_PY, 'r', encoding='utf-8') as f:
    content = f.read()

print("✅ Read main.py")

# Strategy: Remove everything from any @app.get("/api/properties") 
# until we hit the next complete endpoint or major structure

# Pattern 1: Complete broken endpoints
pattern1 = r'@app\.get\("/api/properties"\)\s*\nasync def get_properties.*?(?=\n@app\.|if __name__|class [A-Z])'

# Pattern 2: Orphaned async def without decorator
pattern2 = r'\nasync def get_properties\(skip.*?(?=\n@app\.|if __name__|class [A-Z]|^\ndef [a-z])'

# Pattern 3: Just the broken function
pattern3 = r'def get_properties\(skip.*?(?=\n@app\.|if __name__|class [A-Z]|^\ndef [a-z])'

print("\nRemoving broken endpoints...")

# Apply pattern 1
matches1 = re.findall(pattern1, content, re.DOTALL | re.MULTILINE)
print(f"Pattern 1 matches: {len(matches1)}")
content = re.sub(pattern1, '', content, flags=re.DOTALL | re.MULTILINE)

# Apply pattern 2
matches2 = re.findall(pattern2, content, re.DOTALL | re.MULTILINE)
print(f"Pattern 2 matches: {len(matches2)}")
content = re.sub(pattern2, '', content, flags=re.DOTALL | re.MULTILINE)

# Apply pattern 3
matches3 = re.findall(pattern3, content, re.DOTALL | re.MULTILINE)
print(f"Pattern 3 matches: {len(matches3)}")
content = re.sub(pattern3, '', content, flags=re.DOTALL | re.MULTILINE)

# Now check for any remaining "get_properties" functions
remaining = content.count('def get_properties(')
print(f"\nRemaining get_properties functions: {remaining}")

if remaining > 0:
    print("⚠️  Still have broken functions. Manual inspection needed.")
    # Find them
    for i, line in enumerate(content.split('\n'), 1):
        if 'def get_properties' in line:
            print(f"  Line {i}: {line}")

# Now add ONE clean endpoint after /api/properties/raw
raw_pos = content.find('@app.get("/api/properties/raw")')
if raw_pos == -1:
    print("❌ Cannot find /api/properties/raw")
    exit(1)

# Find next @app decorator
next_app = content.find('\n@app.', raw_pos + 10)
if next_app == -1:
    print("❌ Cannot find insertion point")
    exit(1)

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

final = content[:next_app] + CLEAN_ENDPOINT + content[next_app:]

# Write
with open(MAIN_PY, 'w', encoding='utf-8') as f:
    f.write(final)

print("\n✅ Wrote cleaned file")

# Final verification
with open(MAIN_PY, 'r', encoding='utf-8') as f:
    verify = f.read()

count = verify.count('@app.get("/api/properties")')
has_async = verify.count('async def get_properties(') 
has_log = '🔍 GET /api/properties called' in verify

print("\n" + "="*80)
print("FINAL VERIFICATION")
print("="*80)
print(f"@app.get('/api/properties') count: {count}")
print(f"async def get_properties count: {has_async}")
print(f"Has debug log: {has_log}")

if count == 1 and has_async == 1 and has_log:
    print("\n✅✅✅ CLEAN! File should work now!")
else:
    print("\n❌ Still issues. Run diagnose_syntax_error.py to inspect.")