import redis
import json

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

# Check Redis connection
print("Redis connection:", redis_client.ping())

# Retrieve data from Redis
food_json = redis_client.get("food_data")

# Check whether data exists
if food_json is None:
    print("food_data not found in Redis")
else:
    # Convert JSON string back to Python dictionary
    food_data = json.loads(food_json)

    print("Food data retrieved successfully")
    print(food_data)