from flask import Flask, request, jsonify, render_template
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModel
import torch
import random
from googletrans import Translator

app = Flask(__name__)

# === GLOBAL VARIABLES ===
translator = Translator(service_urls=['translate.google.com'])
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
    "animal": ["Dog", "Cat", "Elephant", "Lion", "Tiger", "Cow", "Monkey", "Rabbit", "Horse","goat"],
    "bird": ["Sparrow", "Pigeon", "Parrot", "Crow", "Peacock", "Duck", "Owl", "Eagle", "Penguin"],
    "flower": ["Rose", "Sunflower", "Lily", "Lotus", "Daisy", "Marigold", "Hibiscus", "Tulip"],
    "vehicle": ["Car", "Bus", "Bicycle", "Train", "Aeroplane", "Boat", "Motorcycle", "Truck"],
    "fruit": ["Apple", "Banana", "Mango", "Orange", "Grapes", "Pineapple", "Watermelon", "Strawberry"],
    "vegetable": ["Potato", "Tomato", "Carrot", "Onion", "Spinach", "Cucumber", "Peas", "Broccoli"],
    "colors": ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Black", "White", "Brown"],
    "clothes": ["Shirt", "Pants", "Dress", "Hat", "Shoes", "Socks", "Jacket", "Scarf", "Gloves", "Belt"],
    "weather": ["Rain", "Snow", "Sunny", "Windy", "Cloudy", "Storm", "Hail", "Fog", "Rainbow", "Lightning"],
    "jobs": ["Doctor", "Teacher", "Farmer", "Pilot", "Chef", "Police", "Artist", "Scientist", "Driver", "Firefighter"],
    "sports": ["Football", "Cricket", "Tennis", "Basketball", "Hockey", "Badminton", "Swimming", "Volleyball", "Baseball", "Cycling"],
    "body parts": ["Hand", "Leg", "Head", "Eye", "Ear", "Nose", "Mouth", "Arm", "Foot", "Neck"],
    "insects": ["Ant", "Bee", "Butterfly", "Spider", "Mosquito", "Cockroach", "Grasshopper", "Beetle", "Ladybug", "Fly"],
}

def random_species(category):
    return random.choice(species[category.lower()])

def get_random_english_word(category):
    """
    Selects a random word from the given category and generates hints as definitions from WordNet.
    """
    word = random_species(category)  # Randomly select a word from the category
    synsets = wn.synsets(word)  # Fetch all synsets (meanings) for the word
    if synsets:
        hints = [synset.definition() for synset in synsets[:2]]  # Take up to 2 definitions as hints
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

# === TRANSLATION FUNCTION ===
def translate_page_content(content, lang):
    """
    Translates the given dictionary of page content into the selected language.
    """
    translated_content = {}
    for key, text in content.items():
        try:
            translated_content[key] = translator.translate(text, dest=lang).text
        except Exception as e:
            translated_content[key] = text  # Fallback to original text if translation fails
    return translated_content

# === ROUTES ===
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/subcategories", methods=["GET"])
def get_subcategories():
    """
    Returns the list of words (subcategories) for a selected category.
    """
    category = request.args.get("category", "").strip().lower()
    if category in species:
        return jsonify({"subcategories": species[category]})
    else:
        return jsonify({"error": "Invalid category"}), 400

@app.route("/translate", methods=["POST"])
def translate():
    """
    Translates the static content of the page into the selected language.
    """
    data = request.json
    language_code = data.get("language")

    # Define static content for translation
    content = {
        "title": "Wordle Game",
        "category_label": "Select Category:",
        "guess_label": "Select Your Guess:",
        "submit_guess": "Submit Guess",
        "hint_button": "Hint",
        "reset_button": "Reset Game",
        "show_button": "Show Word",
    }

    # Translate the content
    translated_content = translate_page_content(content, language_code)
    return jsonify(translated_content)

@app.route("/guess", methods=["POST"])
def guess_word():
    global game_state
    data = request.json
    guess = data["guess"].strip().lower()
    category = data["category"].strip().lower()

    # Initialize game state if not done already
    if not game_state["target_word"]:
        word_info = get_random_english_word(category)
        game_state["target_word"], game_state["hints"] = word_info[0].lower(), word_info[1]

    target_word = game_state["target_word"]

    # Calculate similarity score
    target_emb = get_word_embedding(target_word)
    guess_emb = get_word_embedding(guess)
    similarity_score = cosine_similarity(target_emb, guess_emb)

    # Case-insensitive match
    if guess == target_word:
        game_state["game_over"] = True
        return jsonify({
            "message": f"Congratulations! You've guessed the word '{target_word}'!",
            "success": True,
            "similarity": similarity_score
        })

    # Increment attempts
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
    if game_state["hints_used"] < 2:
        hint = game_state["hints"][game_state["hints_used"]]
        game_state["hints_used"] += 1
        return jsonify({"hint": hint})
    else:
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
        return jsonify({"word": game_state["target_word"]})
    else:
        return jsonify({"word": "No word has been chosen yet."})

if __name__ == "__main__":
    app.run(debug=True)
