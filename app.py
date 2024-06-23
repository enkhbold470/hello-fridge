from flask import Flask, jsonify, request, render_template
import os
import google.generativeai as genai
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
# Configure Google AI with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Upload function for Gemini
def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


# Create the Generative Model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Function to capture image
app.config["UPLOAD_FOLDER"] = "uploads/"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)
        print(image_path, "file uploaded")
        if image_path:
            file = upload_to_gemini(image_path, mime_type="image/jpeg")

            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            file,
                            "What are the food in this image? list all the food or drinks in the image and estimate the weight. List them like this - A can of soda, likely between 12 and 16 oz, or about 350-475 grams. Add the category for those food.",
                        ],
                    },
                ]
            )

            response = chat_session.send_message(
                "What are the food in this image? list all the food or drinks in the image and estimate the weight. List them like this - A can of soda, likely between 12 and 16 oz, or about 350-475 grams. Add the category for those food."
            )

            processed_response = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            response.text,
                            "Based on the given information only get the name, weight, category no other stuff.",
                        ],
                    },
                ]
            ).send_message(
                "Based on the given information only get the name, weight, category no other stuff."
            )

            return jsonify({"response": processed_response.text})

        return jsonify({"error": "Failed to capture image"}), 500


@app.route("/recipe", methods=["POST"])
def generate_recipe():
    # Ensure that an image file is uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(image_path)
    print(f"Uploaded image saved as {image_path}")

    # Upload image to Gemini
    uploaded_file = upload_to_gemini(image_path, mime_type="image/jpeg")

    # Start chat session to identify food in the image
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    uploaded_file,
                    "What are the food in this image? List all the food or drinks in the image and estimate the weight. List them like this - A can of soda, likely between 12 and 16 oz, or about 350-475 grams. Add the category for those food.",
                ],
            },
        ]
    )

    # Retrieve response identifying food and their details
    response = chat_session.send_message(
        "What are the food in this image? List all the food or drinks in the image and estimate the weight. List them like this - A can of soda, likely between 12 and 16 oz, or about 350-475 grams. Add the category for those food."
    )

    # Start another chat session to process the response and generate a recipe prompt
    processed_response = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    response.text,
                    "Based on the given information only get the name, weight, category. No other stuff.",
                ],
            },
        ]
    ).send_message(
        "Based on the given information only get the name, weight, category. No other stuff."
    )

    # Generate a recipe prompt based on identified food
    recipe_prompt = f"Generate a recipe using {processed_response.text}."

    # Start a chat session to generate the recipe
    recipe_generation = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    processed_response.text,
                    recipe_prompt,
                ],
            },
        ]
    )

    # Retrieve the generated recipe
    recipe_response = recipe_generation.send_message(recipe_prompt)

    return jsonify({"recipe": recipe_response.text})


@app.route("/update-sensor", methods=["POST"])
def update_sensor():
    data = request.json
    ecost = 0
    ecost_CA = 0.19
    # print(data)
    ecost = ecost + (ecost_CA * (data["watt"] / 1000)) / 3600
    month = (ecost_CA * (data["watt"] / 1000)) * 720
    return jsonify(
        {
            "message": "Sensor data updated",
            "watt": data["watt"],
            "ecost": ecost,
            "month": month,
        }
    )


if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(host="0.0.0.0", port=5000, debug=True)
