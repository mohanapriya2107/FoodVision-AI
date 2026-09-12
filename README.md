# 🍽️ FoodVision AI — AI-Based Food Recognition & Nutrition System

> An end-to-end deep learning application that recognizes food from images, compares multiple CNN-based models, predicts the food category with confidence, and retrieves corresponding nutritional information using Redis.

---

## 📌 Project Overview

**FoodVision AI** is an end-to-end image classification system designed to recognize food items from uploaded images.

The project combines:

- 🧠 Deep Learning
- 🖼️ Image Classification
- 📊 Model Evaluation
- ⚡ Redis-based Data Retrieval
- 🌐 Flask REST Backend
- 💻 HTML, CSS & JavaScript Frontend
- 🥗 Nutrition Information Retrieval

The system supports **34 different food categories** and evaluates three different deep learning approaches:

1. **Custom CNN**
2. **VGG16**
3. **ResNet50**

Users can upload a food image through the web interface, select a trained model, receive the predicted food class and confidence score, and view nutritional information associated with the prediction.

---

## 🎯 Objectives

The main objectives of FoodVision AI are to:

- Build an image classification system for multiple food categories.
- Train and evaluate different CNN architectures.
- Compare Custom CNN, VGG16, and ResNet50 performance.
- Provide food predictions through a web application.
- Retrieve nutritional information using Redis.
- Display prediction confidence and model performance.
- Create a complete end-to-end ML deployment workflow.

---

## ✨ Key Features

### 🖼️ Food Image Classification
Upload a food image and identify the corresponding food category.

### 🧠 Multiple Deep Learning Models

The application supports:

| Model | Description |
|---|---|
| Custom CNN | CNN architecture designed specifically for the project |
| VGG16 | Transfer-learning based convolutional architecture |
| ResNet50 | Deep residual neural network |

### 📊 Model Performance

The application can display evaluation metrics including:

- Accuracy
- Precision
- Recall
- F1-Score
- Support

### 🥗 Nutrition Information

After identifying a food item, the system retrieves associated nutrition information stored in Redis.

Depending on the available food entry, information can include:

- Protein
- Fat
- Carbohydrates
- Sodium
- Cholesterol

### ⚡ Redis Integration

Redis is used as the fast data layer for:

- Food information
- Nutrition data
- Model performance metrics

### 🌐 Web Application

The project provides a Flask-based backend and browser-based frontend for interacting with the trained models.

### 📱 User-Friendly Interface

The interface provides:

- Food image upload
- Model selection
- Prediction result
- Confidence score
- Uploaded image preview
- Predicted food image
- Nutrition information
- Model performance metrics

---

# 🍕 Supported Food Classes

The system supports **34 food categories**:

1. Baked Potato
2. Crispy Chicken
3. Donut
4. Fries
5. Hot Dog
6. Sandwich
7. Taco
8. Taquito
9. Apple Pie
10. Burger
11. Butter Naan
12. Chai
13. Chapati
14. Cheesecake
15. Chicken Curry
16. Chole Bhature
17. Dal Makhani
18. Dhokla
19. Fried Rice
20. Ice Cream
21. Idli
22. Jalebi
23. Kaathi Rolls
24. Kadai Paneer
25. Kulfi
26. Masala Dosa
27. Momos
28. Omelette
29. Paani Puri
30. Pakode
31. Pav Bhaji
32. Pizza
33. Samosa
34. Sushi

---

# 📂 Dataset

The project uses a structured image dataset containing **34 food classes**.

### Dataset Distribution

Each class contains:

| Dataset | Images per Class | Total Images |
|---|---:|---:|
| Training | 250 | 8,500 |
| Validation | 20 | 680 |
| Testing | 50 | 1,700 |
| **Total** | **320** | **10,880** |

### Dataset Split

```text
Total Images: 10,880

                 Food Dataset
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       Training    Validation    Testing
       8,500         680         1,700
         │            │            │
      250/class     20/class     50/class
