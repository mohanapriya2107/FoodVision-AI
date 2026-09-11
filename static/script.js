// =========================================================
// FOOD CLASSIFICATION FRONTEND
// =========================================================

// Selected model
let selectedModel = "cnn";

// Uploaded image
let selectedImage = null;


// =========================================================
// DOM ELEMENTS
// =========================================================

const imageInput = document.getElementById("imageInput");
const chooseImageButton = document.getElementById("chooseImageButton");

const previewImage = document.getElementById("previewImage");
const photoText = document.getElementById("photoText");

const predictButton = document.getElementById("predictButton");

const loading = document.getElementById("loading");
const errorMessage = document.getElementById("errorMessage");

const classList = document.getElementById("classList");

const actualClassSelect = document.getElementById("actualClass");


// =========================================================
// PAGE LOAD
// =========================================================

document.addEventListener("DOMContentLoaded", function () {

    loadFoodClasses();

    setupModelButtons();

    setupImageUpload();

});


// =========================================================
// IMAGE UPLOAD
// =========================================================

function setupImageUpload() {

    if (!chooseImageButton || !imageInput) {
        return;
    }

    chooseImageButton.addEventListener("click", function () {
        imageInput.click();
    });


    imageInput.addEventListener("change", function (event) {

        const file = event.target.files[0];

        if (!file) {
            return;
        }


        // Check image
        if (!file.type.startsWith("image/")) {

            showError("Please select a valid image file.");

            imageInput.value = "";

            return;
        }


        selectedImage = file;


        // Preview
        const reader = new FileReader();

        reader.onload = function (e) {

            previewImage.src = e.target.result;

            previewImage.style.display = "block";

            if (photoText) {
                photoText.style.display = "none";
            }

        };

        reader.readAsDataURL(file);


        hideError();

    });

}


// =========================================================
// MODEL BUTTONS
// =========================================================

function setupModelButtons() {

    const modelButtons =
        document.querySelectorAll(".model-button");


    modelButtons.forEach(function (button) {

        button.addEventListener("click", function () {

            modelButtons.forEach(function (btn) {
                btn.classList.remove("active");
            });


            button.classList.add("active");


            selectedModel =
                button.dataset.model;


            console.log(
                "Selected model:",
                selectedModel
            );

        });

    });

}


// =========================================================
// LOAD FOOD CLASSES FROM REDIS THROUGH FLASK
// =========================================================

async function loadFoodClasses() {

    try {

        if (classList) {

            classList.innerHTML =
                `<div class="class-loading">
                    Loading classes...
                </div>`;

        }


        const response =
            await fetch("/food-classes");


        const data =
            await response.json();


        console.log(
            "Food classes response:",
            data
        );


        if (!response.ok || !data.success) {

            throw new Error(
                data.error ||
                "Could not load food classes."
            );

        }


        const classes =
            data.classes || [];


        if (classes.length === 0) {

            throw new Error(
                "No food classes found in Redis."
            );

        }


        displayFoodClasses(classes);

        populateActualClass(classes);


    }
    catch (error) {

        console.error(
            "Class loading error:",
            error
        );


        if (classList) {

            classList.innerHTML =
                `<div class="class-loading">
                    Unable to load food classes.
                    <br>
                    <small>${escapeHtml(error.message)}</small>
                </div>`;

        }

    }

}


// =========================================================
// DISPLAY FOOD CLASSES
// =========================================================

function displayFoodClasses(classes) {

    if (!classList) {
        return;
    }


    classList.innerHTML = "";


    classes.forEach(function (foodClass, index) {

        const item =
            document.createElement("div");

        item.className = "class-item";


        item.innerHTML = `
            <div class="class-number">
                ${index + 1}
            </div>

            <span>
                ${escapeHtml(foodClass)}
            </span>
        `;


        classList.appendChild(item);

    });

}


// =========================================================
// POPULATE ACTUAL CLASS DROPDOWN
// =========================================================

function populateActualClass(classes) {

    if (!actualClassSelect) {
        return;
    }


    actualClassSelect.innerHTML =
        `<option value="">
            Select actual class
        </option>`;


    classes.forEach(function (foodClass) {

        const option =
            document.createElement("option");

        option.value = foodClass;

        option.textContent = foodClass;

        actualClassSelect.appendChild(option);

    });

}


// =========================================================
// PREDICT BUTTON
// =========================================================

if (predictButton) {

    predictButton.addEventListener(
        "click",
        predictFood
    );

}


// =========================================================
// PREDICT FOOD
// =========================================================

async function predictFood() {

    hideError();


    // -----------------------------------------------------
    // Check image
    // -----------------------------------------------------

    if (!selectedImage) {

        showError(
            "Please choose a food image first."
        );

        return;
    }


    // -----------------------------------------------------
    // User details
    // -----------------------------------------------------

    const nameInput =
        document.getElementById("name");

    const designationInput =
        document.getElementById("designation");


    const name =
        nameInput
            ? nameInput.value.trim()
            : "";


    const designation =
        designationInput
            ? designationInput.value.trim()
            : "";


    // -----------------------------------------------------
    // Actual class
    // -----------------------------------------------------

    let actualClass = "";


    if (actualClassSelect) {

        actualClass =
            actualClassSelect.value.trim();

    }


    // -----------------------------------------------------
    // FormData
    // -----------------------------------------------------

    const formData =
        new FormData();


    formData.append(
        "image",
        selectedImage
    );


    formData.append(
        "model",
        selectedModel
    );


    formData.append(
        "actual_class",
        actualClass
    );


    formData.append(
        "name",
        name
    );


    formData.append(
        "designation",
        designation
    );


    // -----------------------------------------------------
    // Loading
    // -----------------------------------------------------

    setLoading(true);


    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        console.log(
            "Prediction response:",
            data
        );


        if (!response.ok || !data.success) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );

        }


        // -------------------------------------------------
        // Store result for results.html
        // -------------------------------------------------

        const resultData = {

            name: name,

            designation: designation,

            model: data.model,

            predicted_class:
                data.predicted_class,

            predicted_index:
                data.predicted_index,

            actual_class:
                data.actual_class,

            predicted_metrics:
                data.predicted_metrics || {},

            actual_metrics:
                data.actual_metrics || {},

            nutrition:
                data.nutrition || {}

        };


        sessionStorage.setItem(
            "predictionResult",
            JSON.stringify(resultData)
        );


        // -------------------------------------------------
        // Store uploaded image for results page
        // -------------------------------------------------

        const reader =
            new FileReader();


        reader.onload = function () {

            sessionStorage.setItem(
                "uploadedImage",
                reader.result
            );


            window.location.href =
                "/results";

        };


        reader.readAsDataURL(
            selectedImage
        );


    }
    catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        showError(
            error.message ||
            "Prediction failed."
        );

    }
    finally {

        setLoading(false);

    }

}


// =========================================================
// LOADING STATE
// =========================================================

function setLoading(isLoading) {

    if (loading) {

        if (isLoading) {

            loading.classList.remove(
                "hidden"
            );

        }
        else {

            loading.classList.add(
                "hidden"
            );

        }

    }


    if (predictButton) {

        predictButton.disabled =
            isLoading;

    }

}


// =========================================================
// ERROR MESSAGE
// =========================================================

function showError(message) {

    if (!errorMessage) {
        alert(message);
        return;
    }


    errorMessage.textContent =
        message;


    errorMessage.classList.remove(
        "hidden"
    );

}


function hideError() {

    if (!errorMessage) {
        return;
    }


    errorMessage.textContent = "";

    errorMessage.classList.add(
        "hidden"
    );

}


// =========================================================
// HTML ESCAPE
// =========================================================

function escapeHtml(value) {

    const div =
        document.createElement("div");

    div.textContent =
        value == null ? "" : value;

    return div.innerHTML;

}