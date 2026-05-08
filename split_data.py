import json
import os

# Create the target folder
os.makedirs("raw_data", exist_ok=True)

# 1. Load your master file
print("Reading master file...")
with open("kalkulio_all.json", "r", encoding="utf-8") as f:
    all_houses = json.load(f)

# 2. Loop through the array and save each house individually
for i, house in enumerate(all_houses):
    # Try to use the original ID for the filename, otherwise use a number
    house_id = house.get("metadata", {}).get("original_id", str(i+1))
    filename = f"raw_data/house_{house_id}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(house, f, indent=2, ensure_ascii=False)

print(f"✅ Success! Split {len(all_houses)} houses into the 'raw_data' folder.")