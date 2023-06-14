from flask import Flask, render_template, request,jsonify

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

data = pd.read_csv("myanimelist.csv")
data["genre"] = data["genre"].apply(lambda x: x.strip("[]").replace("'", "").split(", "))
data = data[["uid", "title", "genre","img_url","link","synopsis"]]
data = data.fillna("")

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(data["genre"])

train_data, test_data = train_test_split(genre_matrix, test_size=0.2, random_state=42)

autoencoder = MLPRegressor(hidden_layer_sizes=(128, 64, 128), max_iter=200, random_state=42)
autoencoder.fit(train_data, train_data)

def recommend_anime_with_similarity(genres):
    genres_encoded = mlb.transform([genres])
    encoded_prediction = autoencoder.predict(genres_encoded)
    decoded_prediction = autoencoder.predict(encoded_prediction)
    similarity_scores = (genre_matrix * decoded_prediction).sum(axis=1)
    top_indices = similarity_scores.argsort()[-10:][::-1]
    top_similarity_scores = similarity_scores[top_indices]
    return data.iloc[top_indices], top_similarity_scores

def calculate_accuracy(recommended_animes, user_genres):
    correct_predictions = 0
    total_predictions = 0
    for index, row in recommended_animes.iterrows():
        predicted_genres = set(row["genre"])
        true_genres = set(user_genres)
        correct_predictions += len(predicted_genres.intersection(true_genres))
        total_predictions += len(predicted_genres)

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy


app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

@app.route("/", methods=["GET", "POST"])
def index():
    all_genres = sorted(list({'Fantasy', 'Ecchi', 'Horror', 'Kids', 'Thriller', 'Police', 'Vampire', 'Mecha', 'Sports', 'Psychological', 'Adventure', 'Super Power', 'Josei', 'Demons', 'Cars', 'Game', 'Yuri', 'Historical', 'Shounen', 'Military', 'Dementia', 'Martial Arts', 'Seinen', 'Drama', 'Shoujo', 'Music', 'Supernatural', 'Hentai', 'Shounen Ai', 'Comedy', 'Action', 'Sci-Fi', 'Romance', 'Harem', 'Parody', 'Space', 'Samurai', 'School', 'Yaoi', 'Shoujo Ai', 'Mystery', 'Slice of Life', 'Magic'}))
    return render_template("index.html", all_genres=all_genres)

@app.route("/recommend", methods=["POST"])
def recommend():
    user_genres = request.form.get("user_genres").split(", ")
    recommended_animes, similarity_scores = recommend_anime_with_similarity(user_genres)
    accuracy = calculate_accuracy(recommended_animes, user_genres)
    results = recommended_animes.to_dict(orient="records")
    return jsonify(results=results, accuracy=accuracy)
if __name__ == "__main__":
    app.run(debug=True)
