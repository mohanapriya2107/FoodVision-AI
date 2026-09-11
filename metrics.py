import redis
import json
import os

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

metrics_files = {
    "cnn_metrics": "performance/custom_cnn_metrics.json",
    "vgg16_metrics": "performance/vgg16_metrics.json",
    "resnet50_metrics": "performance/resnet50_metrics.json"
}

for redis_key, file_path in metrics_files.items():
    try:
        # Check file exists
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            continue
        # Read JSON file
        with open(file_path,"r",encoding="utf-8") as file:
            metrics_data = json.load(file)
        redis_client.set(redis_key,json.dumps(metrics_data))
        print(f"{redis_key} stored successfully")
    except Exception as e:
        print(f"[ERROR] Could not store "f"{redis_key}: {e}")
print("\nAll available model metrics stored in Redis.")