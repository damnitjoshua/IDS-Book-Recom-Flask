from flask import Flask
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Rating count more than 50
csv_url = "https://drive.google.com/file/d/1tIxt00bOAPEKRkc57uuBhPZGDLBwsvJc/view?usp=share_link"
csv_url = 'https://drive.google.com/uc?id=' + csv_url.split('/')[-2]
RatingCountDF = pd.read_csv(csv_url)
RatingCountDFPivot = RatingCountDF.pivot(
    index='ISBN', columns='UserID', values='Rating').fillna(0)


@app.route('/')
def home():
    return 'Hello, World!'


@app.route('/random/<count>')
def random(count):
    books = []

    for i in range(int(count)):
        query_index = np.random.choice(RatingCountDF.shape[0])
        book = RatingCountDF.iloc[query_index]
        books.append({
            "title": book['Title'],
            "author": book['Author'],
            "isbn": book['ISBN'],
            "image": book['Image'],
            "category": book["Category"][2:-2],
            "rating": str(book['Rating']),
            "ratingCount": str(book["RatingCount"]),
            "yearOfPublication": str(book["YearOfPublication"])
        })

    return books


@app.route('/knn/<ISBN>')
def knn(ISBN):
    model = pickle.load(open('model/knn_bookRecom_model.sav', 'rb'))
    search = RatingCountDFPivot.loc[ISBN]

    distances, indices = model.kneighbors(
        search.values.reshape(1, -1), n_neighbors=6)

    books = []

    for i in range(0, len(distances.flatten())):
        if i != 0:
            book = RatingCountDF.iloc[indices.flatten()[i]]
            books.append({
                "title": book['Title'],
                "isbn": book['ISBN'],
                "image": book['Image'],
                "distance": distances.flatten()[i]
            })

    return books[::-1]
