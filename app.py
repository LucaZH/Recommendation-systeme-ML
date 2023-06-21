from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_md")
# Load data and preprocess
data = pd.read_csv("myanimelist.csv")
data["genre"] = data["genre"].apply(lambda x: x.strip("[]").replace("'", "").split(", "))
data = data[["uid", "title", "genre", "img_url", "link", "synopsis"]]
data = data.fillna("")

# Create models and fit
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(data["genre"])

train_data, test_data = train_test_split(genre_matrix, test_size=0.2, random_state=42)

autoencoder = MLPRegressor(hidden_layer_sizes=(512, 256, 128, 64, 128, 256, 512), max_iter=200, random_state=42)
autoencoder.fit(train_data, train_data)

vectorizer = TfidfVectorizer(stop_words='english', tokenizer=lambda text: [token.text for token in nlp(text)])
synopsis_matrix = vectorizer.fit_transform(data["synopsis"])

# Define recommendation functions
def recommend_anime_with_similarity(genres):
    genres_encoded = mlb.transform([genres])
    encoded_prediction = autoencoder.predict(genres_encoded)
    decoded_prediction = autoencoder.predict(encoded_prediction)
    similarity_scores = (genre_matrix * decoded_prediction).sum(axis=1)
    top_indices = similarity_scores.argsort()[-10:][::-1]
    top_similarity_scores = similarity_scores[top_indices]
    return data.iloc[top_indices], top_similarity_scores

def recommend_anime_based_on_description(description):
    description_doc = nlp(description)
    description_vector = np.mean([word.vector for word in description_doc], axis=0).reshape(1, -1)
    similarity_scores = cosine_similarity(description_vector, synopsis_matrix).flatten()
    top_indices = similarity_scores.argsort()[-10:][::-1]
    top_similarity_scores = similarity_scores[top_indices]
    return data.iloc[top_indices], top_similarity_scores

def recommend_anime_combined(genres, description):
    genre_based_animes, genre_based_scores = recommend_anime_with_similarity(genres)
    description_based_animes, description_based_scores = recommend_anime_based_on_description(description)
    
    combined_animes = genre_based_animes.merge(description_based_animes, on='uid', how='inner')
    combined_scores = genre_based_scores * description_based_scores
    
    sorted_indices = np.argsort(combined_scores)[-10:][::-1]
    return combined_animes.iloc[sorted_indices], combined_scores[sorted_indices]
# Flask app
app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

@app.route("/", methods=["GET", "POST"])
def index():
    all_genres = sorted(list({'Fantasy', 'Ecchi', 'Horror', 'Kids', 'Thriller', 'Police', 'Vampire', 'Mecha', 'Sports', 'Psychological', 'Adventure', 'Super Power', 'Josei', 'Demons', 'Cars', 'Game', 'Yuri', 'Historical', 'Shounen', 'Military', 'Dementia', 'Martial Arts', 'Seinen', 'Drama', 'Shoujo', 'Music', 'Supernatural', 'Hentai', 'Shounen Ai', 'Comedy', 'Action', 'Sci-Fi', 'Romance', 'Harem', 'Parody', 'Space', 'Samurai', 'School', 'Yaoi', 'Shoujo Ai', 'Mystery', 'Slice of Life', 'Magic'}))

    recommended_animes = pd.DataFrame()
    top_similarity_scores = []

    if request.method == "POST":
        user_genres = request.form.getlist("user_genres")
        description = request.form.get("description")

        if user_genres and description:
            recommended_animes, top_similarity_scores = recommend_anime_combined(user_genres, description)
        elif user_genres:
            recommended_animes, top_similarity_scores = recommend_anime_with_similarity(user_genres)
        elif description:
            recommended_animes, top_similarity_scores = recommend_anime_based_on_description(description)

    return render_template("index.html", all_genres=all_genres, recommended_animes=recommended_animes, top_similarity_scores=top_similarity_scores)

if __name__ == "__main__":
    app.run(debug=True)
