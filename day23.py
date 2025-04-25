from flask import Flask, render_template, request
from textblob import TextBlob
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# In-memory data
feedback_data = [
    {"patient": "Alice", "treatment": "DrugA", "feedback": "I felt much calmer and could sleep better after taking this. Highly recommend it!"},
    {"patient": "Bob", "treatment": "DrugA", "feedback": "It helped reduce my anxiety, but I still had trouble sleeping."},
    {"patient": "Charlie", "treatment": "DrugB", "feedback": "I got a terrible rash. Would not recommend."},
    {"patient": "Diana", "treatment": "DrugB", "feedback": "Mild side effects but worked okay for my condition."},
    {"patient": "Alice", "treatment": "DrugC", "feedback": "Had no noticeable impact after a week."},
    {"patient": "Eve", "treatment": "DrugC", "feedback": "Felt more energetic and less foggy in the morning."},
    {"patient": "Bob", "treatment": "DrugC", "feedback": "I didn’t feel any different. Maybe it takes more time."},
    {"patient": "You", "treatment": "DrugX", "feedback": "I didn’t like it. It made me dizzy and tired."}
]

def sentiment_to_rating(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.5:
        return 5
    elif polarity > 0.1:
        return 4
    elif polarity > -0.1:
        return 3
    elif polarity > -0.5:
        return 2
    else:
        return 1

def get_recommendations(patient_name, data):
    df = pd.DataFrame(data)
    df["rating"] = df["feedback"].apply(sentiment_to_rating)

    pivot = df.pivot_table(index="patient", columns="treatment", values="rating", fill_value=0)

    if patient_name not in pivot.index:
        pivot.loc[patient_name] = 0

    similarity = cosine_similarity(pivot)
    sim_df = pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)

    sim_scores = sim_df[patient_name].drop(patient_name)
    similar_patients = sim_scores.sort_values(ascending=False).head(2).index.tolist()

    unrated_treatments = pivot.loc[patient_name][pivot.loc[patient_name] == 0].index.tolist()

    recommendations = {}
    for treatment in unrated_treatments:
        weighted_sum = 0
        sim_total = 0
        for sim_pat in similar_patients:
            sim_score = sim_df.at[patient_name, sim_pat]
            rating = pivot.at[sim_pat, treatment]
            weighted_sum += sim_score * rating
            sim_total += sim_score
        if sim_total > 0 and weighted_sum > 0:
            predicted_rating = weighted_sum / sim_total
        else:
            predicted_rating = 3.0  # fallback default neutral rating
        
        # Add to recommendations dictionary
        recommendations[treatment] = predicted_rating

    # Return sorted recommendations by predicted rating
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        patient = request.form["patient"]
        treatment = request.form["treatment"]
        feedback = request.form["feedback"]

        feedback_data.append({
            "patient": patient,
            "treatment": treatment,
            "feedback": feedback
        })

        recommendations = get_recommendations(patient, feedback_data)

    return render_template("index23.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)


 