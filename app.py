from flask import Flask, jsonify, request, render_template, session
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
import certifi
import ssl
import aiohttp

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
CORS(app)
# Configure Google AI with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
Humeclient = HumeStreamClient(os.getenv("HUME_API_KEY"))

def save_prediction_to_file(prediction):
    with open('data.json', 'w') as json_file:
        json.dump(prediction, json_file, indent=4)
    print("Data has been written to data.json")

async def predict_face_emotion():
    print("called")
    config = FaceConfig(identify_faces=True)
    
    # Create an SSL context that does not verify SSL certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    # Patch aiohttp to use the custom SSL context
    aiohttp.connector.TCPConnector.ssl_context = ssl_context

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
        if image_path:
            file = upload_to_gemini(image_path, mime_type="image/jpeg")

            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            file,
                            "What are the food in this image? list all the food or drinks in the image and estimate the weight. List them like this - A can of soda, likely between 12 and 16 oz, or about 350-475 grams expires at 12th of July. Add the category and exact expiration date by adding the current time plus the average expire time for those food.",
                        ],
                    },
                ]
            )

            response = chat_session.send_message(
                "What are the food in this image? list all the food or drinks in the image and estimate the weight. List them like this - A can of soda, likely between 12 and 16 oz, or about 350-475 grams expires at 12th of July. Add the category and exact expiration date by adding the current time plus the average expire time for those food."
            )

            processed_response = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            response.text,
                            "Based on the given information only get the name, weight, category, and expiration date no other stuff. If there are multiple items, separate them with newlines. Don't add any styles",
                        ],
                    },
                ]
            )
            
            data = processed_response.send_message(
                "Based on the given information only get the name, weight, category, and expiration date no other stuff. If there are multiple items, separate them with newlines. Don't add any styles"
            )

            separated = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [
                            data.text,
                            "Separete the given list of items and make them array of JSON object with fields name, mass, catergory, and expiration_date fields. For example: {name: Apple, weight: 22-30g, catergory: fruit }. Don't give me anything other than the JSON. Don't add any markdown",
                        ],
                    },
                ]
            ).send_message("Separete the given list of items and make them array of JSON object with fields name, mass, catergory, and expiration_date fields. For example: {name: Apple, weight: 22-30g, catergory: fruit }. Don't give me anything other than the JSON. Don't add any markdown")
            
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
        # prediction = {"Angry": 1}
        session['emotion_prediction'] = prediction
        # prediction = asyncio.run(predict_face_emotion())
        return jsonify({"prediction": prediction}), 200
    
    return jsonify({"error": "Failed to capture image"}), 500

@app.route('/recipe', methods=['GET'])
def generate_recipe():
    # Load data from data.json (assuming it's in the same directory)
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data.json: {e}")  # Debug print
        return jsonify({'error': 'Error loading data.json'}), 500

    # Retrieve the emotion prediction from the loaded data
    emotion_prediction = data.get('face')  # Adjust this based on your actual JSON structure

    # Debug print for loaded data
    print(f"Loaded emotion prediction: {emotion_prediction}")

    # Validate the prediction and extract the emotions
    if not emotion_prediction or 'predictions' not in emotion_prediction:
        print("Error: No valid emotion prediction found")  # Debug print
        return jsonify({'error': 'No valid emotion prediction found'}), 400

    # Assuming only one prediction is available in the example JSON
    emotions = emotion_prediction['predictions'][0]['emotions']

    # Debug print for emotions
    print(f"Extracted emotions: {emotions}")

    # Sort emotions by score (descending)
    emotions_sorted = sorted(emotions, key=lambda x: x['score'], reverse=True)

    # Debug print for sorted emotions
    print(f"Sorted emotions: {emotions_sorted}")

    # Take top 5 emotions
    top_5_emotions = emotions_sorted[:5]

    # Extract emotion names from the top 5 emotions
    top_5_emotion_names = [emotion['name'] for emotion in top_5_emotions]

    # Debug print for top 5 emotion names
    print(f"Top 5 emotion names: {top_5_emotion_names}")

    # Convert items_in_fridge to a string
    items_in_fridge_str = json.dumps(items_in_fridge)

    # Generate a recipe prompt based on identified emotions and processed response
    recipe_prompt = f"Based on the top 5 detected emotions: {', '.join(top_5_emotion_names)}, generate a recipe using these items: {items_in_fridge_str}."

    # Debug print for recipe prompt
    print(f"Recipe prompt: {recipe_prompt}")

    # Start a chat session to generate the recipe
    recipe_generation = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    {"text": recipe_prompt}
                ],
            },
        ]
    )

    # Retrieve the generated recipe
    recipe_response = recipe_generation.send_message(recipe_prompt)

    # Debug print for recipe response
    print(f"Generated recipe response: {recipe_response.text}")

    return jsonify({'recipe': recipe_response.text})

electricdata = {
    'ecost': 0,
    'month': 0,
    'watt': 0,
    'message': 'Sensor data updated'
}

@app.route("/update-sensor", methods=["POST"])
def update_sensor():
    global electricdata  # Declare electricdata as global to modify the global variable
    data = request.json
    ecost_CA = 0.19
    ecost = electricdata['ecost'] + (ecost_CA * (data["watt"] / 1000)) / 3600
    month = (ecost_CA * (data["watt"] / 1000)) * 720
    electricdata = {
        "message": "Sensor data updated",
        "watt": data["watt"],
        "ecost": ecost,
        "month": month,
    }
    print(electricdata, "is electric data")
    return jsonify(electricdata)

@app.route("/get-sensor", methods=["GET"])
def get_electric_data():
    print(electricdata, "is electric data")
    return jsonify(electricdata)

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

@app.route('/future-food-plan', methods=['POST'])
def future_food_plan():
    # Ensure that necessary input is provided
    print("uisnasdfasdfasf\n\n\n")
    if not request.json or 'dish_types' not in request.json:
        return jsonify({'error': 'No input provided'}), 400

    dish_types = request.json['dish_types']
    print(dish_types)

    # Generate a prompt to create a food plan for the next 7 days
    food_plan_prompt = (f"Create a food plan for the next 7 days based on the following items in the fridge: "
                        f"{', '.join(item['name'] for item in items_in_fridge)}. "
                        f"Include breakfast, lunch, and dinner for each day. The user prefers the following types of dishes: "
                        f"{', '.join(dish_types)}.")

    # Start a chat session to generate the food plan
    food_plan_generation = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    food_plan_prompt,
                ],
            },
        ]
    )
    food_plan_response = food_plan_generation.send_message(food_plan_prompt)

    return jsonify({'food_plan': food_plan_response.text})



if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(host="0.0.0.0", port=5000, debug=True)
