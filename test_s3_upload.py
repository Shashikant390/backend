import requests
import json

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
# 1. Your Public App URL (from Railway Dashboard)
#    Make sure this is the HTTPS link, NOT the database link.
API_URL = "https://backend-production-59ed.up.railway.app"

# 2. Test Data
ENDPOINT = f"{API_URL}/farms"

# 3. Headers
# Since you have DEV_AUTH="true" (based on your logs), 
# your app likely accepts a simple header to identify the user.
# If you are using Firebase Auth, you would normally put a real token here.
HEADERS = {
    "Content-Type": "application/json",
    # Try one of these depending on how your auth.py is written:
    "Authorization": "Bearer test_token_123", 
    "X-DEV-UID": "test_user_123"  # Common dev bypass
}

# 4. Valid GeoJSON Polygon (A square field in India)
payload = {
    "name": "My First Railway Farm",
    "meta": {"crop": "Wheat", "season": "Rabi"},
    "geojson": {
        "type": "Polygon",
        "coordinates": [[
            [75.906, 17.659],
            [75.908, 17.659],
            [75.908, 17.661],
            [75.906, 17.661],
            [75.906, 17.659]
        ]]
    }
}

def create_farm():
    print(f"🚀 Sending request to: {ENDPOINT}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(ENDPOINT, json=payload, headers=HEADERS)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 201:
            print("\n✅ SUCCESS! Farm created in the cloud DB.")
        elif response.status_code == 401:
            print("\n❌ Auth Failed. Check your Headers or 'get_current_user' logic.")
        else:
            print("\n⚠️ Request failed.")
            
    except Exception as e:
        print(f"\n❌ Connection Error: {e}")

if __name__ == "__main__":
    create_farm()