import redis
import json

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

# Read JSON file
with open("food.json", "r", encoding="utf-8") as file:
    food_data = json.load(file)

# Store entire JSON under ONE key
redis_client.set(
    "food_data",
    json.dumps(food_data)
)

print("Food data stored successfully")