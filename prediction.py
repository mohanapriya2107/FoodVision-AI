import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

CUSTOM_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "custom_cnn.keras"
)

VGG_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "vgg16.keras"
)

RESNET_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "resnet_model.keras"
)


# ============================================================
# CLASS NAMES
# IMPORTANT: KEEP THE SAME ORDER USED DURING TRAINING
# ============================================================

CLASS_NAMES = [
    "Baked Potato",
    "Crispy Chicken",
    "Donut",
    "Fries",
    "Hot Dog",
    "Sandwich",
    "Taco",
    "Taquito",
    "apple_pie",
    "burger",
    "butter_naan",
    "chai",
    "chapati",
    "cheesecake",
    "chicken_curry",
    "chole_bhature",
    "dal_makhani",
    "dhokla",
    "fried_rice",
    "ice_cream",
    "idli",
    "jalebi",
    "kaathi_rolls",
    "kadai_paneer",
    "kulfi",
    "masala_dosa",
    "momos",
    "omelette",
    "paani_puri",
    "pakode",
    "pav_bhaji",
    "pizza",
    "samosa",
    "sushi"
]


# ============================================================
# CHECK MODEL FILE
# ============================================================

def check_model_file(path, model_name):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{model_name} model not found:\n{path}"
        )

    file_size = os.path.getsize(path)

    if file_size < 1000:

        raise ValueError(
            f"{model_name} file is only "
            f"{file_size} bytes.\n"
            f"It may be a Git LFS pointer instead "
            f"of the actual .keras model."
        )

    print(
        f"[OK] {model_name} found "
        f"({file_size:,} bytes)"
    )


# ============================================================
# LOAD SAVED MODELS DIRECTLY
# ============================================================

custom_model = None
vgg_model = None
resnet_model = None


def load_custom_model():

    global custom_model

    if custom_model is None:

        check_model_file(
            CUSTOM_MODEL_PATH,
            "Custom CNN"
        )

        print(
            "Loading saved Custom CNN model..."
        )

        custom_model = load_model(
            CUSTOM_MODEL_PATH,
            compile=False
        )

        print(
            "Custom CNN loaded successfully."
        )

    return custom_model


def load_vgg_model():

    global vgg_model

    if vgg_model is None:

        check_model_file(
            VGG_MODEL_PATH,
            "VGG16"
        )

        print(
            "Loading saved VGG16 model..."
        )

        vgg_model = load_model(
            VGG_MODEL_PATH,
            compile=False
        )

        print(
            "VGG16 loaded successfully."
        )

    return vgg_model


def load_resnet_model():

    global resnet_model

    if resnet_model is None:

        check_model_file(
            RESNET_MODEL_PATH,
            "ResNet50"
        )

        print(
            "Loading saved ResNet50 model..."
        )

        resnet_model = load_model(
            RESNET_MODEL_PATH,
            compile=False
        )

        print(
            "ResNet50 loaded successfully."
        )

    return resnet_model


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    img = image.load_img(
        image_path,
        target_size=(256, 256)
    )

    img_array = image.img_to_array(
        img
    )

    return img_array


# ============================================================
# CUSTOM CNN PREPROCESSING
# ============================================================

def preprocess_custom(image_path):

    img_array = load_image(
        image_path
    )

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# VGG16 PREPROCESSING
# ============================================================

def preprocess_vgg(image_path):

    img_array = load_image(
        image_path
    )

    img_array = tf.keras.applications.vgg16.preprocess_input(
        img_array
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# RESNET50 PREPROCESSING
# ============================================================

def preprocess_resnet(image_path):

    img_array = load_image(
        image_path
    )

    img_array = tf.keras.applications.resnet50.preprocess_input(
        img_array
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ============================================================
# COMMON PREDICTION FUNCTION
# ============================================================

def get_prediction(
    model,
    img_array
):

    predictions = model.predict(
        img_array,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    return predicted_class, confidence


# ============================================================
# CUSTOM CNN PREDICTION
# ============================================================

def predict_custom(image_path):

    model = load_custom_model()

    img_array = preprocess_custom(
        image_path
    )

    return get_prediction(
        model,
        img_array
    )


# ============================================================
# VGG16 PREDICTION
# ============================================================

def predict_vgg(image_path):

    model = load_vgg_model()

    img_array = preprocess_vgg(
        image_path
    )

    return get_prediction(
        model,
        img_array
    )


# ============================================================
# RESNET50 PREDICTION
# ============================================================

def predict_resnet(image_path):

    model = load_resnet_model()

    img_array = preprocess_resnet(
        image_path
    )

    return get_prediction(
        model,
        img_array
    )


# ============================================================
# TEST MODEL LOADING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FoodVision AI - Saved Model Test")
    print("=" * 60)

    print("\nLoading Custom CNN...")
    load_custom_model()

    print("\nLoading VGG16...")
    load_vgg_model()

    print("\nLoading ResNet50...")
    load_resnet_model()

    print("\n" + "=" * 60)
    print("ALL SAVED MODELS LOADED SUCCESSFULLY")
    print("=" * 60)