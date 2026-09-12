# 🍔 FoodVision AI

An intelligent deep learning web application that predicts food items from images, providing real-time food classification, confidence scores, nutritional details, and comprehensive model performance metrics across multiple deep learning architectures.

The system automates the complete computer vision pipeline—from image preprocessing and feature extraction to real-time model inference and Redis-backed nutritional data retrieval.

---

## 📌 Project Overview

Food recognition using computer vision plays a critical role in automated nutrition tracking, dietary management, smart health applications, and food industry automation.

**FoodVision AI** helps users instantly identify food categories from uploaded images while offering deeper insights into nutritional values and deep learning model reliability.

The application is built using:
* Python
* TensorFlow / Keras
* Flask
* Redis
* NumPy & Pandas
* HTML5, CSS3, & JavaScript

---

## 🚀 Features

* 🍽️ **Food Image Classification:** Accurately classifies food items across 34 distinct classes.
* 🤖 **Multi-Model Engine:** Support for Custom CNN, VGG16, and ResNet50 architectures.
* 🎯 **Probability & Confidence Scoring:** Generates confidence scores for every prediction.
* 🥗 **Nutritional Information Retrieval:** Retrieves instant nutritional details using Redis in-memory storage.
* 📊 **Model Performance Metrics:** Live evaluation displays class-wise Precision, Recall, F1-Score, and Accuracy.
* 📷 **Image Upload & Preview:** Supports user image upload with real-time web previews.
* 🖼️ **Dynamic Food Image Mapping:** Displays representative static reference images alongside user uploads.
* 🌐 **Flask Web Interface:** Clean, interactive UI for effortless user interaction.
* ⚡ **Real-time Prediction:** Low-latency inference pipeline built for high responsiveness.

---

## 🍴 Supported Food Classes

FoodVision AI currently supports **34 food categories**:

| | | |
|---|---|---|
| 1. Baked Potato | 13. Chapati | 25. Kulfi |
| 2. Crispy Chicken | 14. Cheesecake | 26. Masala Dosa |
| 3. Donut | 15. Chicken Curry | 27. Momos |
| 4. Fries | 16. Chole Bhature | 28. Omelette |
| 5. Hot Dog | 17. Dal Makhani | 29. Paani Puri |
| 6. Sandwich | 18. Dhokla | 30. Pakode |
| 7. Taco | 19. Fried Rice | 31. Pav Bhaji |
| 8. Taquito | 20. Ice Cream | 32. Pizza |
| 9. Apple Pie | 21. Idli | 33. Samosa |
| 10. Burger | 22. Jalebi | 34. Sushi |
| 11. Butter Naan | 23. Kaathi Rolls | |
| 12. Chai | 24. Kadai Paneer | |

---

## 🧠 Complete Machine Learning Workflow

```text
[ Raw Dataset: 10,880 Images across 34 Classes ]
                     │
                     ▼
       ┌───────────────────────────┐
       │ Data Preprocessing        │
       │ - Resize: 256x256x3       │
       │ - Color Space: RGB        │
       │ - Scaling / Normalization │
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ Multi-Architecture Train  │
       │ - Custom CNN from scratch │
       │ - Transfer: VGG16         │
       │ - Transfer: ResNet50      │
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ Model Serialization       │
       │ - .keras Model Artifacts  │
       │ - Precomputed JSON Metrics│
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ Serving & Caching Engine  │
       │ - Flask HTTP Endpoint     │
       │ - Redis In-Memory Cache   │
       └───────────────────────────┘
```

### 1. Data Collection & Distribution
The system is trained on a structured dataset containing 34 food categories with balanced distribution across partitions:

```text
Training Set   : 8,500 images  (250 images per class)
Validation Set :   680 images  (20 images per class)
Testing Set    : 1,700 images  (50 images per class)
------------------------------------------------------
Total Dataset  : 10,880 images
```

### 2. Data Cleaning & Preprocessing Pipeline
* **Image Resizing:** All input images are normalized to a standardized spatial resolution of $256 \times 256 \times 3$.
* **Color Space Conversion:** Standardized to 3-channel RGB to eliminate alpha channel or grayscale mismatches.
* **Pixel Normalization:** Normalized channel intensities to scale ranges expected by deep backbones (e.g., $[0, 1]$ or ImageNet zero-centered scaling).
* **Dataset Augmentation & Optimization:** Shuffled, batched, and pre-fetched via `tf.data` pipelines to maximize GPU throughput and prevent overfitting.

### 3. Deep Learning Architectures

* **Custom CNN:**
  * Built from scratch using sequential stacks of Convolution 2D ($3\times3$), Batch Normalization, ReLU activation, Max Pooling ($2\times2$), and Dropout ($0.25$ to $0.5$).
  * Dense output projection with Softmax activation over 34 targets.
  * **Saved Model:** `models/custom_cnn.keras`

* **VGG16 Backbone:**
  * Utilizes transfer learning initialized on ImageNet weights.
  * Deep stacked $3\times3$ convolutions extract fine-grained visual primitives (edges, surface oil, color contrast).
  * Feature maps flattened and passed into fully connected dense layers.
  * **Saved Model:** `models/vgg16.keras`

* **ResNet50 Backbone:**
  * Leverages residual identity shortcut connections to solve vanishing gradients across 50 layers.
  * Strong representation learning for complex, multi-ingredient food compositions.
  * **Saved Model:** `models/resnet_model.keras`

---

## 📊 Model Evaluation & Metrics

Each trained model is evaluated against the 1,700 unseen test set images.

* **Accuracy:** Percentage of overall correct classifications across all classes.
* **Precision:** $\frac{TP}{TP + FP}$ — Precision of predicted food classes.
* **Recall:** $\frac{TP}{TP + FN}$ — Sensitivity to ground truth food categories.
* **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

| Model Architecture | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Custom CNN** | *[Add Value]* | *[Add Value]* | *[Add Value]* | *[Add Value]* |
| **VGG16** | *[Add Value]* | *[Add Value]* | *[Add Value]* | *[Add Value]* |
| **ResNet50** | *[Add Value]* | *[Add Value]* | *[Add Value]* | *[Add Value]* |

---

## 💾 Model & File Assets

```text
models/custom_cnn.keras         → Custom CNN Architecture Weights & Pipeline
models/vgg16.keras              → VGG16 Transfer Learning Model
models/resnet_model.keras       → ResNet50 Transfer Learning Model
metrics/custom_cnn_metrics.json → Class-wise Performance Metrics (CNN)
metrics/vgg16_metrics.json      → Class-wise Performance Metrics (VGG16)
metrics/resnet50_metrics.json   → Class-wise Performance Metrics (ResNet50)
data/food.json                  → Food Master File (Nutritional Data)
```

---

## 🌐 End-to-End Inference Workflow

```text
               User Uploads Food Image via Browser
                                │
                                ▼
                 Flask Route Handler (/predict)
                                │
                                ▼
       Target Architecture Selection (CNN / VGG16 / ResNet50)
                                │
                                ▼
            Inference Preprocessing (Resize, RGB, Rescale)
                                │
                                ▼
             TensorFlow Model Inference (.predict())
                                │
                                ▼
           Argmax Mapping → Class Label + Confidence (%)
                                │
                                ▼
           Redis Key-Value In-Memory Lookup (food_data)
                                │
                                ▼
         Fetch Class-wise Metrics (Precision, Recall, F1)
                                │
                                ▼
       Render results.html with Macros, Metrics, and Reference Image
```

---

## 🗃️ Redis In-Memory Caching Setup

Redis serves as an ultra-low latency cache to retrieve static nutritional details and precomputed model evaluation metrics without querying persistent disk files on each inference call.

* **Food Data:** Cached under key `food_data`.
* **Architecture Metrics:** Cached under keys `custom_cnn_metrics`, `vgg16_metrics`, and `resnet50_metrics`.

### Redis Ingestion Script

```python
import json
import redis

# Initialize local Redis client
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Cache Nutritional Master Data
with open("data/food.json", "r", encoding="utf-8") as file:
    food_data = json.load(file)
redis_client.set("food_data", json.dumps(food_data))

# Cache Precomputed Model Metrics
metric_files = {
    "custom_cnn_metrics": "metrics/custom_cnn_metrics.json",
    "vgg16_metrics": "metrics/vgg16_metrics.json",
    "resnet50_metrics": "metrics/resnet50_metrics.json",
}

for key, path in metric_files.items():
    with open(path, "r", encoding="utf-8") as f:
        redis_client.set(key, json.dumps(json.load(f)))

print("All nutritional datasets and model metrics cached to Redis.")
```

---

## 📂 Project Structure

```text
FoodVision-AI
│
├── app.py
├── prediction.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── custom_cnn.keras
│   ├── vgg16.keras
│   └── resnet_model.keras
│
├── metrics/
│   ├── custom_cnn_metrics.json
│   ├── vgg16_metrics.json
│   └── resnet50_metrics.json
│
├── templates/
│   ├── index.html
│   └── results.html
│
├── static/
│   ├── style.css
│   ├── script.js
│   ├── food_images/
│   └── screenshots/
│
└── data/
    └── food.json
```

---

## ⚙️ Technologies Used

* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow 2.x, Keras
* **Computer Vision & Utilities:** OpenCV, Pillow, NumPy, Pandas
* **Backend Web Framework:** Flask
* **In-Memory Store:** Redis
* **Frontend:** HTML5, CSS3, JavaScript
* **Model Serialization:** Keras H5 / `.keras` format via Git LFS

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/mohanapriya2107/FoodVision-AI.git](https://github.com/mohanapriya2107/FoodVision-AI.git)
cd FoodVision-AI
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Redis Service
Ensure Redis is running locally:
```bash
redis-cli ping
# Output: PONG
```

### 5. Run the Application
```bash
python app.py
```

Access the web interface at: `http://127.0.0.1:5000/`

---

## 📈 Prediction Output

For every submitted image, the application generates:
* **Identified Food Category:** (e.g., Samosa, Pizza, Butter Naan)
* **Confidence Score (%):** (e.g., 98.4%)
* **Nutritional Breakdown:** Calories, Carbs, Proteins, Fats, and Fiber
* **Model Performance Metrics:** Class-level Precision, Recall, and F1-Score for validation transparency

---

## 🎯 Future Enhancements

* 📱 Mobile Application Development (Flutter / React Native)
* ☁️ Cloud Containerization & Deployment (Docker, AWS ECS / GCP Cloud Run)
* 📊 Explainable AI via Grad-CAM / SHAP Heatmaps
* 🥗 Personalized Dietary & Macronutrient Tracking
* 🎯 Top-$K$ Prediction Probabilities Display
* 🧠 Multi-Model Weighted Ensemble Inference
* 📷 Live Camera Stream Video Classification

---

## 👩‍💻 Author

**Mohana Priya Korukoppula**  
AI & Machine Learning Engineer  
* **GitHub:** [MohanaPriya2107](https://github.com/mohanapriya2107)  
* **LinkedIn:** [Mohana Priya Korukoppula](https://www.linkedin.com/in/mohana-priya-579b04253/)
