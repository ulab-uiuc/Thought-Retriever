import json
import os

# File to store the list
filename = 'thoughts_abs_tradeoff.json'

# Function to load the list of thoughts
def load_thoughts():
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}  # Return an empty dictionary if JSON is invalid
    else:
        return {}
 

# Function to save the list to a file
def save_thoughts(thoughts):
    with open(filename, 'w') as file:
        json.dump(thoughts, file, indent=4)

# Function to add a new title or update an existing one
def add_or_update_thought(title, thoughts, new_thought):
    thoughts[title].append(new_thought)

    return thoughts
