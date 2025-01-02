from flask import Flask, request, jsonify, render_template
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModel
import torch
import random

app = Flask(__name__)

# === GLOBAL VARIABLES ===
game_state = {
    "max_attempts": 3,
    "attempts": 0,
    "hints_used": 0,
    "game_over": False,
    "target_word": None,
    "hints": [],
}

# === WORD GENERATION ===
species = {
    "animal": ["Dog", "Cat", "Elephant", "Lion", "Tiger", "Cow", "Monkey", "Rabbit", "Horse", "Goat"],
    "bird": ["Sparrow", "Pigeon", "Parrot", "Crow", "Peacock", "Duck", "Owl", "Eagle", "Penguin", "Woodpecker"],
    "vehicle": ["Car", "Bus", "Bicycle", "Train", "Aeroplane", "Boat", "Motorcycle", "Truck", "Boat", "Ship"],
    "fruit": ["Apple", "Banana", "Mango", "Orange", "Grapes", "Pineapple", "Watermelon", "Strawberry", "Pomegranate", "Kiwi"],
    "vegetable": ["Potato", "Tomato", "Carrot", "Onion", "Spinach", "Cucumber", "Peas", "Broccoli", "Brinjal", "Ladies Finger"],
    "clothes": ["Shirt", "Pants", "Dress", "Hat", "Shoes", "Socks", "Jacket", "Scarf", "Gloves", "Belt"],
    "weather": ["Rainy", "Snow", "Sunny", "Windy", "Cloudy", "Storm", "Hail", "Fog", "Rainbow", "Lightning"],
    "jobs": ["Doctor", "Teacher", "Farmer", "Pilot", "Chef", "Police", "Artist", "Scientist", "Driver", "Firefighter"],
    "sports": ["Football", "Cricket", "Tennis", "Basketball", "Hockey", "Badminton", "Swimming", "Volleyball", "Baseball", "Cycling"],
    "insects": ["Ant", "Bee", "Butterfly", "Spider", "Mosquito", "Cockroach", "Grasshopper", "Beetle", "Ladybug", "Fly"],
}

def random_species(category):
    return random.choice(species[category.lower()])

def get_random_english_word(category):
    word = random_species(category)
    synsets = wn.synsets(word)
    if synsets:
        hints = [synset.definition() for synset in synsets[:2]]
        return [word, hints]
    return None

# === WORD EMBEDDING MODEL ===
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2).item()

# === ROUTES ===
@app.route("/")
def home():
    return render_template("default.html")

@app.route("/game")
def game():
    return render_template("index.html")


@app.route("/subcategories", methods=["GET"])
def get_subcategories():
    category = request.args.get("category", "").strip().lower()
    if category in species:
        return jsonify({"subcategories": species[category]})
    else:
        return jsonify({"error": "Invalid category"}), 400

@app.route("/guess", methods=["POST"])
def guess_word():
    global game_state
    data = request.json
    guess = data["guess"].strip().lower()
    category = data["category"].strip().lower()

    if not game_state["target_word"]:
        word_info = get_random_english_word(category)
        game_state["target_word"], game_state["hints"] = word_info[0].lower(), word_info[1]

    target_word = game_state["target_word"]

    target_emb = get_word_embedding(target_word)
    guess_emb = get_word_embedding(guess)
    similarity_score = cosine_similarity(target_emb, guess_emb)

    if guess == target_word:
        game_state["game_over"] = True
        return jsonify({
            "message": f"Congratulations! You've guessed the word '{target_word}'!",
            "success": True,
            "similarity": similarity_score
        })

    game_state["attempts"] += 1
    if game_state["attempts"] >= game_state["max_attempts"]:
        game_state["game_over"] = True
        return jsonify({
            "message": f"Game over! The word was '{target_word}'.",
            "success": False,
            "similarity": similarity_score
        })

    return jsonify({
        "message": f"Incorrect guess! You have {game_state['max_attempts'] - game_state['attempts']} attempts remaining.",
        "success": False,
        "similarity": similarity_score
    })

@app.route("/hint", methods=["GET"])
def get_hint():
    global game_state
    if game_state["hints_used"] < len(game_state["hints"]):
        hint = game_state["hints"][game_state["hints_used"]]
        game_state["hints_used"] += 1
        return jsonify({"hint": hint})
    return jsonify({"hint": "No more hints available!"})

@app.route("/reset", methods=["POST"])
def reset_game():
    global game_state
    game_state = {
        "max_attempts": 3,
        "attempts": 0,
        "hints_used": 0,
        "game_over": False,
        "target_word": None,
        "hints": [],
    }
    return jsonify({"message": "Game has been reset!"})

@app.route("/show", methods=["GET"])
def show_word():
    global game_state
    if game_state["target_word"]:
        return jsonify({"word": f"The word was: {game_state['target_word']}"})
    return jsonify({"word": "No word has been chosen yet."})




# Routes for each category
@app.route('/animal.html')
def animal():
    return render_template('animal.html')

@app.route('/birds.html')
def birds():
    return render_template('birds.html')

@app.route('/clothes.html')
def clothes():
    return render_template('clothes.html')

@app.route('/fruits.html')
def fruits():
    return render_template('fruits.html')

@app.route('/insects.html')
def insects():
    return render_template('insects.html')

@app.route('/jobs.html')
def jobs():
    return render_template('jobs.html')

@app.route('/sports.html')
def sports():
    return render_template('sports.html')

@app.route('/vegetables.html')
def vegetables():
    return render_template('vegetables.html')

@app.route('/vehicles.html')
def vehicles():
    return render_template('vehicles.html')

@app.route('/weather.html')
def weather():
    return render_template('weather.html')

if __name__ == "__main__":
    app.run(debug=True)