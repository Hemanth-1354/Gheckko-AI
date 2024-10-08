import nltk
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('punkt_tab')

castle_swimmer_info = {
    "about": "Castle Swimmer is a fantasy webcomic about a young swimmer who discovers the mysteries of a magical underwater world.",
    "main characters": "The main characters include the swimmer named 'Kano' and the mysterious sea creature 'Luno'.",
    "theme": "The story explores themes of friendship, bravery, and the struggle between light and darkness.",
    "prophecy": "In Chapters 83-89, the new prophecy unveils secrets about the fate of the ocean and its inhabitants.",
    "setting": "The story is set in a fantastical underwater kingdom filled with vibrant sea creatures and hidden dangers.",
    "conflict": "The main conflict revolves around Kano's journey to uncover the truth behind the prophecy and protect his world.",
    "genre": "Castle Swimmer is primarily a fantasy webcomic with elements of adventure and drama.",
    "art style": "The webcomic features a beautiful, colorful art style that enhances the magical atmosphere of the story.",
    "author": "Castle Swimmer is created by a talented artist known as 'Jae'.",
}

def get_response(user_input):
    tokens = word_tokenize(user_input.lower())
    
    if "about" in tokens:
        return castle_swimmer_info["about"]
    elif "characters" in tokens or "who" in tokens:
        return castle_swimmer_info["main characters"]
    elif "theme" in tokens:
        return castle_swimmer_info["theme"]
    elif "prophecy" in tokens:
        return castle_swimmer_info["prophecy"]
    elif "setting" in tokens:
        return castle_swimmer_info["setting"]
    elif "conflict" in tokens:
        return castle_swimmer_info["conflict"]
    elif "genre" in tokens:
        return castle_swimmer_info["genre"]
    elif "art style" in tokens:
        return castle_swimmer_info["art style"]
    elif "author" in tokens:
        return castle_swimmer_info["author"]
    else:
        return "I'm sorry, I don't understand that. You can ask me about the plot, main characters, themes, prophecy, setting, conflict, genre, art style, or author."

def chat():
    print("Welcome to the Castle Swimmer Chatbot! You can ask me about the story.")
    print("Type 'exit' to end the chat.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
