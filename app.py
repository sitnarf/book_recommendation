from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello_world():  # put application's code here
    return "Hello World!"


@app.route("/recommend/<isbn>")
def recommend(isbn):
    recommendations = [
        {"title": "Book A", "author": "Author A"},
        {"title": "Book B", "author": "Author B"},
        {"title": "Book C", "author": "Author C"},
    ]
    return {"isbn": isbn, "recommendations": recommendations}


if __name__ == "__main__":
    app.run()
