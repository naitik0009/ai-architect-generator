import json
import os
import random
import copy

INPUT_DIR = "raw_data"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_conversations = []

def round_floats(obj):
    """Recursively rounds floats in the JSON to 2 decimal places to save massive token space."""
    if isinstance(obj, float):
        return round(obj, 2)
    elif isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(v) for v in obj]
    return obj

def format_conversation(floor_plan_json, area_m2):
    """Wraps the JSON in the conversation format MLX needs, stripping fluff and decimals"""
    
    # 1. Keep only what Kalkulio cares about
    cleaned_json = {
        "steny": floor_plan_json.get("steny", []),
        "otvory": floor_plan_json.get("otvory", []),
        "prostory": floor_plan_json.get("prostory", [])
    }
    
    # 2. Round all crazy decimals to 2 places
    cleaned_json = round_floats(cleaned_json)
    
    # 3. Dump the JSON without ANY spaces or newlines (separators=(',', ':'))
    compact_json_str = json.dumps(cleaned_json, separators=(',', ':'))
    
    return {
        "messages": [
            {"role": "system", "content": "You are an expert architectural AI. Generate a valid JSON floor plan for a single-family house."},
            {"role": "user", "content": f"Generate a floor plan for a house with an approximate area of {area_m2}m2."},
            {"role": "assistant", "content": compact_json_str}
        ]
    }

def augment_scale(original_json, scale_factor):
    """Scales coordinates up or down"""
    new_json = copy.deepcopy(original_json)
    for wall in new_json.get("steny", []):
        wall["od"] = [round(wall["od"][0] * scale_factor, 4), round(wall["od"][1] * scale_factor, 4)]
        wall["do"] = [round(wall["do"][0] * scale_factor, 4), round(wall["do"][1] * scale_factor, 4)]
    for space in new_json.get("prostory", []):
        space["polygon"] = [[round(p[0] * scale_factor, 4), round(p[1] * scale_factor, 4)] for p in space["polygon"]]
        space["plocha_m2"] = round(space.get("plocha_m2", 0) * (scale_factor ** 2), 2)
    return new_json

def augment_mirror_x(original_json):
    """Mirrors the house horizontally by negating the X coordinates"""
    new_json = copy.deepcopy(original_json)
    for wall in new_json.get("steny", []):
        wall["od"] = [round(-wall["od"][0], 4), wall["od"][1]]
        wall["do"] = [round(-wall["do"][0], 4), wall["do"][1]]
    for space in new_json.get("prostory", []):
        space["polygon"] = [[round(-p[0], 4), p[1]] for p in space["polygon"]]
    return new_json

def augment_mirror_y(original_json):
    """Mirrors the house vertically by negating the Y coordinates"""
    new_json = copy.deepcopy(original_json)
    for wall in new_json.get("steny", []):
        wall["od"] = [wall["od"][0], round(-wall["od"][1], 4)]
        wall["do"] = [wall["do"][0], round(-wall["do"][1], 4)]
    for space in new_json.get("prostory", []):
        space["polygon"] = [[p[0], round(-p[1], 4)] for p in space["polygon"]]
    return new_json

# 1. Process all files in raw_data
processed_count = 0
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".json"):
        processed_count += 1
        filepath = os.path.join(INPUT_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            house_data = json.load(f)
            total_area = round(sum([p.get("plocha_m2", 0) for p in house_data.get("prostory", [])]), 1)
            
            # Original
            all_conversations.append(format_conversation(house_data, total_area))
            # Scale Up (+10%)
            all_conversations.append(format_conversation(augment_scale(house_data, 1.1), round(total_area * 1.21, 1)))
            # Scale Down (-10%)
            all_conversations.append(format_conversation(augment_scale(house_data, 0.9), round(total_area * 0.81, 1)))
            # Mirror X
            all_conversations.append(format_conversation(augment_mirror_x(house_data), total_area))
            # Mirror Y
            all_conversations.append(format_conversation(augment_mirror_y(house_data), total_area))

# 2. Shuffle and Split
random.shuffle(all_conversations)
split_idx = int(len(all_conversations) * 0.9)
train_data = all_conversations[:split_idx]
valid_data = all_conversations[split_idx:]

# 3. Save
with open(os.path.join(OUTPUT_DIR, "train.jsonl"), 'w', encoding='utf-8') as f:
    for item in train_data: f.write(json.dumps(item) + '\n')

with open(os.path.join(OUTPUT_DIR, "valid.jsonl"), 'w', encoding='utf-8') as f:
    for item in valid_data: f.write(json.dumps(item) + '\n')

print(f"✅ Processed {processed_count} original files.")
print(f"✅ Created {len(train_data)} training and {len(valid_data)} validation examples!")