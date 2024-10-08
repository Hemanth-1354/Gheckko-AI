from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

data = [
    {"description": "A love story set in high school with lots of drama.", "category": "romance"},
    {"description": "A prince falls in love with a commoner and must choose between love and duty.", "category": "romance"},
    {"description": "A young couple navigates the challenges of a long-distance relationship.", "category": "romance"},
    {"description": "A heartwarming tale of love and friendship during a summer vacation.", "category": "romance"},
    {"description": "A romantic comedy about two people who hate each other but fall in love.", "category": "romance"},
    {"description": "A historical romance set in Victorian England.", "category": "romance"},
    {"description": "A warrior fights monsters in a fantasy realm to save his kingdom.", "category": "action"},
    {"description": "A superhero must save the world from a supervillain with a powerful weapon.", "category": "action"},
    {"description": "A young hero embarks on a quest to rescue a kidnapped princess.", "category": "action"},
    {"description": "An epic battle between good and evil that spans multiple worlds.", "category": "action"},
    {"description": "A skilled assassin takes on a dangerous mission to eliminate a corrupt leader.", "category": "action"},
    {"description": "A soldier returns home after a war to face new challenges.", "category": "action"},
    {"description": "A detective solving mysteries in a futuristic city.", "category": "mystery"},
    {"description": "A thrilling journey of a spy infiltrating a criminal organization.", "category": "mystery"},
    {"description": "A young detective uncovers a conspiracy in her town.", "category": "mystery"},
    {"description": "An investigation that reveals hidden secrets in a small town.", "category": "mystery"},
    {"description": "A journalist investigates a series of unsolved crimes.", "category": "mystery"},
    {"description": "A mysterious disappearance leads to a shocking revelation.", "category": "mystery"},
    {"description": "A young girl discovers she has magical powers in a fantasy world.", "category": "fantasy"},
    {"description": "A dragon and a knight must team up to stop a dark wizard.", "category": "fantasy"},
    {"description": "A magical academy trains young wizards to battle dark forces.", "category": "fantasy"},
    {"description": "A journey through a magical realm filled with fantastic creatures.", "category": "fantasy"},
    {"description": "An adventure to find a legendary artifact in a mystical land.", "category": "fantasy"},
    {"description": "A fantasy epic where heroes rise against an ancient evil.", "category": "fantasy"},
    {"description": "A slice of life story following a group of friends navigating high school.", "category": "slice of life"},
    {"description": "A story about family, love, and growing up in a small town.", "category": "slice of life"},
    {"description": "The daily adventures of a quirky group of roommates living together.", "category": "slice of life"},
    {"description": "A coming-of-age story about teenagers facing life’s challenges.", "category": "slice of life"},
    {"description": "A journey of self-discovery during a summer road trip.", "category": "slice of life"},
    {"description": "A heartfelt tale of friendships formed in a local cafe.", "category": "slice of life"},
    {"description": "A young detective solves a complex mystery in a haunted mansion.", "category": "mystery"},
    {"description": "In a dystopian future, a rebel leader fights against an oppressive regime to restore freedom.", "category": "action"},
    {"description": "A story of survival in a post-apocalyptic world filled with danger.", "category": "action"},
    {"description": "A personal diary of a girl navigating the ups and downs of life.", "category": "slice of life"},
    {"description": "A young artist finds inspiration and love in an unexpected place.", "category": "romance"},
    {"description": "A thrilling heist involving an elite team of thieves.", "category": "action"},
]


descriptions = [item['description'] for item in data]
categories = [item['category'] for item in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descriptions)

classifier = DecisionTreeClassifier()

scores = cross_val_score(classifier, X, categories, cv=3)

print("Cross-validation scores: ", scores)
print("Mean accuracy: ", np.mean(scores))

classifier.fit(X, categories)

y_pred = classifier.predict(X)

print("\nClassification Report:")
print(classification_report(categories, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(categories, y_pred))


new_description1 = ["A young mage discovers an ancient spellbook that unlocks a hidden world of mythical creatures, embarking on an epic quest to save her realm from an awakening dark sorcerer."]
new_description2 = ["A group of friends navigates the ups and downs of daily life while working together at a local café, each facing their own personal challenges and growth."]
new_description = new_description1 + new_description2
new_description_vectorized = vectorizer.transform(new_description)
predicted_category = classifier.predict(new_description_vectorized)

print(f"\nPredicted category for new description1: {predicted_category[0]}")
print(f"\nPredicted category for new description2: {predicted_category[1]}")
