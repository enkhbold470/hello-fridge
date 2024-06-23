from flask import Flask, jsonify, request, render_template
import os
import google.generativeai as genai
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import re
import json
import asyncio
from hume import HumeStreamClient, StreamSocket
from hume.models.config import FaceConfig

load_dotenv()
ecost_CA = 0.19

app = Flask(__name__)
CORS(app)
# Configure Google AI with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
Humeclient = HumeStreamClient(os.getenv("HUME_API_KEY"))

def save_prediction_to_file(prediction):
    with open('data.json', 'w') as json_file:
        json.dump(prediction, json_file, indent=4)
    print("Data has been written to data.json")

async def predict_face_emotion() :
    print("called")
    config = FaceConfig(identify_faces=True)
    async with Humeclient.connect([config]) as socket:
        result = await socket.send_file("face.jpg")
        save_prediction_to_file(result)
        return result   


# Upload function for Gemini
def upload_to_gemini(path, mime_type=None):
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def split_by_non_alphanumeric(s: str):
    # Regular expression pattern to match non-alphanumeric characters
    pattern = r'[^a-zA-Z0-9]+'
    # Split the string using the pattern
    substrings = re.split(pattern, s)
    return substrings

items_in_fridge = []

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
                            "Based on the given information only get the name, weight, category no other stuff. If there are multiple items, separate them with newlines. Don't add any styles",
                        ],
                    },
                ]
            )
            
            data = processed_response.send_message(
                "Based on the given information only get the name, weight, category no other stuff. If there are multiple items, separate them with newlines. Don't add any styles"
            )

            separated = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            data.text,
                            "Separete the given list of items and make them array of JSON object with fields name, mass, and catergory fields. For example: {name: Apple, weight: 22-30g, catergory: fruit }. Don't give me anything other than the JSON. Don't add any markdown",
                        ],
                    },
                ]
            ).send_message("Separete the given list of items and make them array of JSON object with fields name, mass, and catergory fields. For example: {name: Apple, weight: 22-30g, catergory: fruit }. Don't give me anything other than the JSON. Don't add any markdown")
            
            # splitted_names = split_by_non_alphanumeric(separated.text)
            item_list = json.loads(separated.text)
            for item in item_list:
                items_in_fridge.append(item)

            # print(splitted_names)
            return jsonify({"response": item_list})

        return jsonify({"error": "Failed to capture image"}), 500


@app.route("/hume", methods=["POST"])
def faceRecognition():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file.save("face.jpg")
        print("face.jpg", "file uploaded")
        prediction = asyncio.run(predict_face_emotion())
        return jsonify({"prediction": prediction}), 200
    
    return jsonify({"error": "Failed to capture image"}), 500



@app.route('/recipe', methods=['POST'])
def generate_recipe():
    # Ensure that the processed response text is provided
    if not request.json or 'processed_response' not in request.json:
        return jsonify({'error': 'No processed response provided'}), 400
    
    processed_response = request.json['processed_response']
    
    # Generate a recipe prompt based on identified food
    recipe_prompt = f"Generate a recipe using {processed_response}."
    
    # Start a chat session to generate the recipe
    recipe_generation = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    processed_response,
                    recipe_prompt,
                ],
            },
        ]
    )
    
    # Retrieve the generated recipe
    recipe_response = recipe_generation.send_message(recipe_prompt)
    
    return jsonify({'recipe': recipe_response.text})


ecost = 0


@app.route("/update-sensor", methods=["POST"])
def update_sensor():
    data = request.json
    print(data)
    ecost = ecost + (ecost_CA * (data["watt"] / 1000)) / 3600
    month = (ecost_CA * (data["watt"] / 1000)) * 720
    return jsonify(
        {
            "message": "Sensor data updated",
            "watt": data["watt"],
            "ecost": round(ecost, 2),
            "month": round(month, 2),
        }
    )


@app.route('/alexa', methods=['GET'])
def get_concatenated_names():
    # Extract the 'name' fields of all objects in target_array
    names = [item['name'] for item in items_in_fridge]

    # Concatenate names with "and" before the last element and " are here" at the end
    if len(names) > 1:
        concatenated_names = ', '.join(names[:-1]) + ' and ' + names[-1] + ' are here'
    else:
        concatenated_names = names[0] + ' are here' if names else 'No items are here'

    # Add the prefix "In the fridge"
    concatenated_names_with_prefix = 'In the fridge, ' + concatenated_names

    return jsonify(concatenated_names_with_prefix)


if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(host="0.0.0.0", port=5000, debug=True)
