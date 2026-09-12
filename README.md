🍔 FoodVision AIAn intelligent deep learning web application that predicts food items from images, providing real-time food classification, confidence scores, nutritional details, and comprehensive model performance metrics across multiple deep learning architectures.The system automates the complete computer vision pipeline—from image preprocessing and feature extraction to real-time model inference and Redis-backed nutritional data retrieval.📌 Project OverviewFood recognition using computer vision plays a critical role in automated nutrition tracking, dietary management, smart health applications, and food industry automation.FoodVision AI helps users instantly identify food categories from uploaded images while offering deeper insights into nutritional values and deep learning model reliability.The application is built using:PythonTensorFlow / KerasFlaskRedisNumPyPandasHTMLCSSJavaScript🚀 Features🍽️ Food Image Classification: Accurately classifies food items across 34 distinct classes.🤖 Multi-Model Engine: Support for Custom CNN, VGG16, and ResNet50 architectures.🎯 Probability &amp; Confidence Scoring: Generates confidence scores for every prediction.🥗 Nutritional Information Retrieval: Retrieves instant nutritional details using Redis in-memory storage.📊 Model Performance Metrics: Live evaluation displays class-wise Precision, Recall, F1-Score, and Accuracy.📷 Image Upload &amp; Preview: Supports user image upload with real-time web previews.🖼️ Dynamic Food Image Mapping: Displays representative static reference images alongside user uploads.🌐 Flask Web Interface: Clean, interactive UI for effortless user interaction.⚡ Real-time Prediction: Low-latency inference pipeline built for high responsiveness.🍴 Supported Food ClassesFoodVision AI currently supports 34 food categories:Plaintext1.  Baked Potato          13. Chapati               25. Kulfi
2.  Crispy Chicken        14. Cheesecake            26. Masala Dosa
3.  Donut                 15. Chicken Curry         27. Momos
4.  Fries                 16. Chole Bhature         28. Omelette
5.  Hot Dog               17. Dal Makhani           29. Paani Puri
6.  Sandwich              18. Dhokla                30. Pakode
7.  Taco                  19. Fried Rice            31. Pav Bhaji
8.  Taquito               20. Ice Cream             32. Pizza
9.  Apple Pie             21. Idli                  33. Samosa
10. Burger                22. Jalebi                34. Sushi
11. Butter Naan           23. Kaathi Rolls
12. Chai                  24. Kadai Paneer
🧠 Machine Learning Workflow1. Data CollectionThe system is trained on a structured dataset containing 34 food categories with a balanced distribution across sets:PlaintextTraining Set   : 8,500 images  (250 per class)
Validation Set :   680 images  (20 per class)
Testing Set    : 1,700 images  (50 per class)
----------------------------------------------
Total Dataset  : 10,880 images
2. Data Cleaning &amp; PreprocessingImage Resizing: Rescaled to standardized input sizes (256 × 256 × 3).Color Space Conversion: Standardized to RGB format across all input images.Pixel Normalization: Normalized pixel values to standard ranges suited for transfer learning backbones.Batch Preparation: Grouped into batches with shuffle and prefetch pipelines for efficient GPU training.

          
            
          
        
  
        
    

3. Deep Learning ArchitecturesCustom CNNDesigned from scratch with alternating Convolutional, ReLU activation, MaxPooling, and Dropout layers.Optimized for lightweight deployment and quick local inference.Saved Model: models/custom_cnn.kerasVGG16Transfer learning setup utilizing deep stacked 3x3 convolutional blocks pre-trained on ImageNet.Captures rich visual features ranging from basic edges to complex visual food textures.Saved Model: models/vgg16.kerasResNet50Deep Residual Network leveraging skip connections to prevent vanishing gradients during deep feature extraction.Achieves high top-1 accuracy on nuanced food categories.Saved Model: models/resnet_model.keras📊 Model Evaluation &amp; MetricsEach trained model is rigorously evaluated on the test set (1,700 unseen images). The system tracks the following key classification metrics:Accuracy: Percentage of overall correct classifications.Precision: Ratio of correctly predicted positive observations to total predicted positives.Recall: Ratio of correctly predicted positive observations to all observations in actual class.F1 Score: Weighted average of Precision and Recall.Model ArchitectureAccuracyPrecisionRecallF1 ScoreCustom CNN[Add Value][Add Value][Add Value][Add Value]VGG16[Add Value][Add Value][Add Value][Add Value]ResNet50[Add Value][Add Value][Add Value][Add Value]💾 Model &amp; File AssetsPlaintextmodels/custom_cnn.keras         → Custom CNN Architecture Weights &amp; Pipeline
models/vgg16.keras              → VGG16 Transfer Learning Model
models/resnet_model.keras       → ResNet50 Transfer Learning Model
metrics/custom_cnn_metrics.json → Class-wise Performance Metrics (CNN)
metrics/vgg16_metrics.json      → Class-wise Performance Metrics (VGG16)
metrics/resnet50_metrics.json   → Class-wise Performance Metrics (ResNet50)
data/food.json                  → Food Master File (Nutritional Data)
🌐 Deployment &amp; Prediction WorkflowPlaintext       User Uploads Image
               │
               ▼
   Selected Deep Learning Model
    (Custom CNN / VGG16 / ResNet50)
               │
               ▼
   Image Validation &amp; Resizing
               │
               ▼
      Pixel Normalization
               │
               ▼
      Model Inference Engine
               │
               ▼
   Predicted Food Class + Confidence Score
               │
               ▼
 Redis In-Memory Lookup (food_data &amp; metrics)
               │
               ▼
 Comprehensive Prediction &amp; Nutrition Output
🗃️ Redis IntegrationRedis serves as an in-memory key-value store providing instant access to nutrition data and model metrics without repeated disk read latency.Food Data Store: Stored under key food_data as JSON.Model Performance Store: Stored under keys custom_cnn_metrics, vgg16_metrics, and resnet50_metrics.Loading Data into RedisPythonimport redis
import json

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Load Food Data
with open("data/food.json", "r", encoding="utf-8") as file:
    food_data = json.load(file)
redis_client.set("food_data", json.dumps(food_data))

print("Data loaded into Redis successfully.")
📂 Project StructurePlaintextFoodVision-AI
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
⚙️ Technologies UsedProgramming Language: PythonDeep Learning: TensorFlow, Keras (CNN, VGG16, ResNet50)Data Processing: NumPy, PandasBackend: FlaskIn-Memory Store: RedisFrontend: HTML5, CSS3, JavaScriptVersion Control: Git, Git LFS (for .keras files)📦 Installation &amp; Setup1. Clone the RepositoryBashgit clone https://github.com/mohanapriya2107/FoodVision-AI.git
cd FoodVision-AI


          
            
          
        
  
        
    

2. Set Up Virtual EnvironmentBashpython -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
3. Install DependenciesBashpip install -r requirements.txt
4. Start Redis ServerVerify Redis is active locally:Bashredis-cli ping
# Output: PONG


          
            
          
        
  
        
    

5. Run the ApplicationBashpython app.py
Open your browser and navigate to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)📈 OutputThe application returns:Identified Food Category (e.g., Samosa, Pizza, Butter Naan)Confidence Score (%) (e.g., 98.4%)Nutritional Breakdown: Calories, Carbs, Proteins, Fats, and FiberModel Class Performance: Precision, Recall, and F1-Score for the predicted class🎯 Future Enhancements📱 Mobile Application Development☁️ Cloud Deployment (AWS / GCP / Heroku)📊 SHAP / Grad-CAM Visual Explainability (Heatmaps)🥗 Personalized Dietary &amp; Macro Recommendations🎯 Top-K Prediction Probabilities Display🧠 Ensemble Model Predictions📷 Live Camera Recognition Support👩‍💻 AuthorMohana Priya KorukoppulaAI &amp; Machine Learning EngineerGitHub: MohanaPriya2107LinkedIn: Mohana Priya Korukoppula
