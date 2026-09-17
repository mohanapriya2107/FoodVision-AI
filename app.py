import os
import json
import uuid

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

from werkzeug.utils import secure_filename

from PIL import Image

from prediction import (
    predict_custom,
    predict_vgg,
    predict_resnet
)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# FOLDERS
# =========================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "data"
)

FOOD_IMAGE_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "food_images"
)


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# Maximum uploaded file size = 10 MB

app.config[
    "MAX_CONTENT_LENGTH"
] = 10 * 1024 * 1024


# =========================================================
# ALLOWED IMAGE TYPES
# =========================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def allowed_file(filename):

    extension = os.path.splitext(
        filename
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# =========================================================
# FOOD CLASS NAMES
# =========================================================

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


# =========================================================
# JSON LOADING
# =========================================================

def load_json_file(filename):

    file_path = os.path.join(
        DATA_FOLDER,
        filename
    )

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        print(
            f"[OK] Loaded JSON: {filename}"
        )

        return data

    except FileNotFoundError:

        print(
            f"[ERROR] JSON file not found: "
            f"{file_path}"
        )

        return {}

    except json.JSONDecodeError as error:

        print(
            f"[ERROR] Invalid JSON: {filename}"
        )

        print(error)

        return {}

    except Exception as error:

        print(
            f"[ERROR] Could not load "
            f"{filename}: {error}"
        )

        return {}


# =========================================================
# LOAD DATA
# =========================================================

food_data = load_json_file(
    "food_Data.json"
)

custom_cnn_metrics = load_json_file(
    "custom_cnn_metrics.json"
)

vgg16_metrics = load_json_file(
    "vgg16_metrics.json"
)

resnet50_metrics = load_json_file(
    "resnet50_metrics.json"
)


MODEL_METRICS = {

    "cnn": custom_cnn_metrics,

    "vgg16": vgg16_metrics,

    "resnet50": resnet50_metrics

}


# =========================================================
# NUTRITION KEY MAPPING
# =========================================================

NUTRITION_KEY_MAP = {

    "Baked Potato":
        "baked_potato",

    "Crispy Chicken":
        "crispy_chicken",

    "Donut":
        "donut",

    "Fries":
        "fries",

    "Hot Dog":
        "hot_dog",

    "Sandwich":
        "sandwich",

    "Taco":
        "taco",

    "Taquito":
        "taquito",

    "apple_pie":
        "apple_pie",

    "burger":
        "burger",

    "butter_naan":
        "butter_naan",

    "chai":
        "chai",

    "chapati":
        "chapati",

    "cheesecake":
        "cheese_cake",

    "chicken_curry":
        "chicken_curry",

    "chole_bhature":
        "chole_bature",

    "dal_makhani":
        "dal_makhani",

    "dhokla":
        "dhokla",

    "fried_rice":
        "fried_rice",

    "ice_cream":
        "ice_cream",

    "idli":
        "idly",

    "jalebi":
        "jalebi",

    "kaathi_rolls":
        "kaathi_rolls",

    "kadai_paneer":
        "kadai_paneer",

    "kulfi":
        "kulfi",

    "masala_dosa":
        "masala_dosa",

    "momos":
        "momos",

    "omelette":
        "omlette",

    "paani_puri":
        "paani_puri",

    "pakode":
        "pakode",

    "pav_bhaji":
        "pav_bhaaji",

    "pizza":
        "pizza",

    "samosa":
        "samosa",

    "sushi":
        "sushi"

}


# =========================================================
# NORMALIZE CLASS NAME
# =========================================================

def normalize_class_name(class_name):

    if not class_name:

        return None

    class_name = str(
        class_name
    ).strip()

    for name in CLASS_NAMES:

        if name.lower() == class_name.lower():

            return name

    return class_name


# =========================================================
# NUTRITION KEY
# =========================================================

def get_nutrition_key(class_name):

    class_name = normalize_class_name(
        class_name
    )

    if not class_name:

        return None

    if class_name in NUTRITION_KEY_MAP:

        return NUTRITION_KEY_MAP[
            class_name
        ]

    return (
        class_name
        .lower()
        .replace(" ", "_")
    )


# =========================================================
# GENERIC JSON LOOKUP
# =========================================================

def find_in_json(
    data,
    target
):

    """
    Recursively searches dictionaries/lists
    for a key matching the target.
    """

    if data is None:

        return None

    target = str(
        target
    ).strip().lower()


    # -----------------------------------------------------
    # DICTIONARY
    # -----------------------------------------------------

    if isinstance(data, dict):

        for key, value in data.items():

            key_text = str(
                key
            ).strip().lower()

            if key_text == target:

                return value


        for value in data.values():

            result = find_in_json(
                value,
                target
            )

            if result is not None:

                return result


    # -----------------------------------------------------
    # LIST
    # -----------------------------------------------------

    elif isinstance(data, list):

        for item in data:

            result = find_in_json(
                item,
                target
            )

            if result is not None:

                return result


    return None


# =========================================================
# CLASS METRICS
# =========================================================

def get_class_metrics(
    model_name,
    class_name
):

    model_name = str(
        model_name
    ).lower().strip()

    class_name = normalize_class_name(
        class_name
    )

    metrics_data = MODEL_METRICS.get(
        model_name,
        {}
    )

    if not metrics_data:

        return {}


    # -----------------------------------------------------
    # EXACT CLASS
    # -----------------------------------------------------

    result = find_in_json(
        metrics_data,
        class_name
    )

    if isinstance(result, dict):

        return result


    # -----------------------------------------------------
    # ALTERNATIVE CLASS NAMES
    # -----------------------------------------------------

    alternative_names = [

        class_name.lower(),

        class_name.lower().replace(
            " ",
            "_"
        )

    ]


    for name in alternative_names:

        result = find_in_json(
            metrics_data,
            name
        )

        if isinstance(result, dict):

            return result


    return {}


# =========================================================
# NUTRITION DATA
# =========================================================

def get_nutrition(class_name):

    nutrition_key = get_nutrition_key(
        class_name
    )

    if not nutrition_key:

        return {}


    # -----------------------------------------------------
    # MAPPED KEY
    # -----------------------------------------------------

    result = find_in_json(
        food_data,
        nutrition_key
    )

    if isinstance(result, dict):

        return result


    # -----------------------------------------------------
    # CLASS NAME
    # -----------------------------------------------------

    result = find_in_json(
        food_data,
        class_name
    )

    if isinstance(result, dict):

        return result


    return {}


# =========================================================
# FOOD IMAGE MAP
# =========================================================

FOOD_IMAGE_MAP = {

    "Baked Potato":
        "baked_potato",

    "Crispy Chicken":
        "crispy_chicken",

    "Donut":
        "donuts",

    "Fries":
        "fries",

    "Hot Dog":
        "hot_dog",

    "Sandwich":
        "sandwich",

    "Taco":
        "taco",

    "Taquito":
        "taquitos",

    "apple_pie":
        "apple_pie",

    "burger":
        "burger",

    "butter_naan":
        "butter_naan",

    "chai":
        "chai",

    "chapati":
        "chapati",

    "cheesecake":
        "cheese_cake",

    "chicken_curry":
        "chicken_curry",

    "chole_bhature":
        "chole_bature",

    "dal_makhani":
        "dal_makhani",

    "dhokla":
        "dhokla",

    "fried_rice":
        "fried_rice",

    "ice_cream":
        "ice_cream",

    "idli":
        "idly",

    "jalebi":
        "jalebi",

    "kaathi_rolls":
        "kaathi_rolls",

    "kadai_paneer":
        "kadai_paneer",

    "kulfi":
        "kulfi",

    "masala_dosa":
        "masala_dosa",

    "momos":
        "momos",

    "omelette":
        "omlette",

    "paani_puri":
        "paani_puri",

    "pakode":
        "pakode",

    "pav_bhaji":
        "pav_bhaaji",

    "pizza":
        "pizza",

    "samosa":
        "samosa",

    "sushi":
        "sushi"

}


# =========================================================
# FOOD IMAGE URL
# =========================================================

def get_food_image_url(class_name):

    image_name = FOOD_IMAGE_MAP.get(
        class_name
    )

    if not image_name:

        return None


    extensions = [

        ".jpg",
        ".jpeg",
        ".JPG",
        ".JPEG",
        ".png",
        ".PNG",
        ".webp",
        ".WEBP"

    ]


    for extension in extensions:

        filename = (
            image_name +
            extension
        )

        file_path = os.path.join(
            FOOD_IMAGE_FOLDER,
            filename
        )

        if os.path.isfile(
            file_path
        ):

            return (
                "/static/food_images/"
                + filename
            )


    return None


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# RESULTS
# =========================================================

@app.route("/results")
def results():

    return render_template(
        "results.html"
    )


# =========================================================
# UPLOADED FILE
# =========================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# =========================================================
# FOOD CLASSES
# =========================================================

@app.route(
    "/food-classes",
    methods=["GET"]
)
def food_classes():

    return jsonify({

        "success": True,

        "classes": CLASS_NAMES,

        "count": len(
            CLASS_NAMES
        )

    })


# =========================================================
# NUTRITION API
# =========================================================

@app.route(
    "/nutrition/<path:food_class>",
    methods=["GET"]
)
def nutrition(food_class):

    try:

        food_class = normalize_class_name(
            food_class
        )

        nutrition_data = get_nutrition(
            food_class
        )

        if not nutrition_data:

            return jsonify({

                "success": False,

                "error":
                    "Food nutrition data not found"

            }), 404


        return jsonify({

            "success": True,

            "food": food_class,

            "nutrition": nutrition_data

        })


    except Exception as error:

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    image_path = None

    try:

        # =================================================
        # IMAGE
        # =================================================

        if "image" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "No image uploaded."

            }), 400


        uploaded_file = request.files[
            "image"
        ]


        if not uploaded_file.filename:

            return jsonify({

                "success": False,

                "error":
                    "No image selected."

            }), 400


        # =================================================
        # FILE TYPE
        # =================================================

        if not allowed_file(
            uploaded_file.filename
        ):

            return jsonify({

                "success": False,

                "error":
                    "Only JPG, JPEG, PNG and WEBP "
                    "images are allowed."

            }), 400


        # =================================================
        # MODEL
        # =================================================

        model_name = request.form.get(
            "model",
            "cnn"
        ).lower().strip()


        if model_name not in [

            "cnn",
            "vgg16",
            "resnet50"

        ]:

            return jsonify({

                "success": False,

                "error":
                    "Invalid model selected."

            }), 400


        # =================================================
        # ACTUAL CLASS
        # =================================================

        actual_class = request.form.get(
            "actual_class",
            ""
        ).strip()


        if actual_class:

            actual_class = normalize_class_name(
                actual_class
            )


        # =================================================
        # SECURE FILENAME
        # =================================================

        original_filename = secure_filename(
            uploaded_file.filename
        )


        extension = os.path.splitext(
            original_filename
        )[1].lower()


        unique_filename = (
            str(uuid.uuid4())
            + extension
        )


        image_path = os.path.join(
            UPLOAD_FOLDER,
            unique_filename
        )


        # =================================================
        # SAVE IMAGE
        # =================================================

        uploaded_file.save(
            image_path
        )


        # =================================================
        # VERIFY IMAGE
        # =================================================

        try:

            with Image.open(
                image_path
            ) as img:

                img.verify()


        except Exception:

            if os.path.exists(
                image_path
            ):

                os.remove(
                    image_path
                )


            return jsonify({

                "success": False,

                "error":
                    "The uploaded file is not a valid image."

            }), 400


        # =================================================
        # PREDICTION
        # =================================================

        print(
            f"[INFO] Prediction requested: "
            f"model={model_name}"
        )


        if model_name == "cnn":

            prediction_result = predict_custom(
                image_path
            )

        elif model_name == "vgg16":

            prediction_result = predict_vgg(
                image_path
            )

        else:

            prediction_result = predict_resnet(
                image_path
            )


        # =================================================
        # HANDLE PREDICTION RESULT
        #
        # prediction.py returns:
        #
        # (predicted_class, confidence)
        #
        # Example:
        #
        # ("pizza", 0.9534)
        # =================================================

        if not isinstance(
            prediction_result,
            tuple
        ):

            raise ValueError(
                "Prediction function must return "
                "(predicted_class, confidence)."
            )


        if len(
            prediction_result
        ) == 0:

            raise ValueError(
                "Prediction returned an empty tuple."
            )


        predicted_class = str(
            prediction_result[0]
        ).strip()


        if len(
            prediction_result
        ) > 1:

            confidence = float(
                prediction_result[1]
            )

        else:

            confidence = 0.0


        # =================================================
        # NORMALIZE PREDICTED CLASS
        # =================================================

        predicted_class = normalize_class_name(
            predicted_class
        )


        # =================================================
        # FIND CLASS INDEX
        # =================================================

        if predicted_class not in CLASS_NAMES:

            raise ValueError(
                "Predicted class not found in "
                "CLASS_NAMES: "
                + str(predicted_class)
            )


        predicted_index = CLASS_NAMES.index(
            predicted_class
        )


        # =================================================
        # CONFIDENCE
        # =================================================

        if confidence <= 1:

            confidence_percentage = (
                confidence * 100
            )

        else:

            confidence_percentage = confidence


        confidence_percentage = max(
            0,
            min(
                100,
                confidence_percentage
            )
        )


        # =================================================
        # METRICS
        # =================================================

        predicted_metrics = get_class_metrics(
            model_name,
            predicted_class
        )


        actual_metrics = {}


        if actual_class:

            actual_metrics = get_class_metrics(
                model_name,
                actual_class
            )


        # =================================================
        # NUTRITION
        # =================================================

        nutrition_data = get_nutrition(
            predicted_class
        )


        # =================================================
        # FOOD IMAGE
        # =================================================

        food_image_url = get_food_image_url(
            predicted_class
        )


        # =================================================
        # RESPONSE
        # =================================================

        response_data = {

            "success": True,

            "model": model_name,

            "predicted_class":
                predicted_class,

            "predicted_index":
                predicted_index,

            "confidence":
                round(
                    confidence_percentage,
                    2
                ),

            "actual_class":
                actual_class,

            "predicted_metrics":
                predicted_metrics,

            "actual_metrics":
                actual_metrics,

            "nutrition":
                nutrition_data,

            "food_image":
                food_image_url,

            "uploaded_image":
                "/uploads/"
                + unique_filename

        }


        print(
            "[OK] Prediction successful:"
        )

        print(
            f"    Model: {model_name}"
        )

        print(
            f"    Class: {predicted_class}"
        )

        print(
            f"    Confidence: "
            f"{confidence_percentage:.2f}%"
        )


        return jsonify(
            response_data
        )


    except Exception as error:

        print(
            "========================================"
        )

        print(
            "PREDICTION ERROR:"
        )

        print(
            repr(error)
        )

        print(
            "========================================"
        )


        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "running",

        "database":
            "JSON files",

        "number_of_classes":
            len(CLASS_NAMES),

        "food_data_loaded":
            bool(food_data),

        "custom_cnn_metrics_loaded":
            bool(custom_cnn_metrics),

        "vgg16_metrics_loaded":
            bool(vgg16_metrics),

        "resnet50_metrics_loaded":
            bool(resnet50_metrics),

        "models": [

            "cnn",
            "vgg16",
            "resnet50"

        ]

    })


# =========================================================
# ERROR HANDLER - FILE TOO LARGE
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "success": False,

        "error":
            "File is too large. "
            "Maximum size is 10 MB."

    }), 413


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    print(
        "FoodVision AI starting..."
    )


    print(
        "Classes:",
        len(CLASS_NAMES)
    )


    print(
        "Data folder:",
        DATA_FOLDER
    )


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )