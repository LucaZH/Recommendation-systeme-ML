import pandas as pd
from ast import literal_eval

df = pd.read_csv("myanimelist.csv")

df['genre'] = df['genre'].apply(literal_eval)

unique_genres = set()
for genres in df['genre']:
    unique_genres.update(genres)

print(unique_genres)
