import pandas as pd
import numpy as np
import flask 
from flask import Flask, request, render_template
import spacy

# Load the SciSpaCy model
nlp = spacy.load("en_core_sci_sm")

app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    entities = []
    if request.method == "POST":
        text = request.form["text"]
        doc = nlp(text)
        entities = doc.ents
    return render_template('index12.html', text=text, entities=entities)

if __name__ == "__main__":
    app.run(debug=True)