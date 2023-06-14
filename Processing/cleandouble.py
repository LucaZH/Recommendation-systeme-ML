import pandas as pd

df = pd.read_csv('anime_dataset.csv')
df.drop_duplicates(subset=['uid', 'title', 'synopsis', 'genre', 'aired', 'episodes', 'members', 'popularity', 'ranked', 'score', 'img_url', 'link'], keep='first', inplace=True)
df.to_csv('cleanedmyanimelist.csv', index=False)

