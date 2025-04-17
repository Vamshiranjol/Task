import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Sample data
data = {
    'disease': [
        'cough','throatpain',
        'sneezing','cold',
        'fever','high temperature',
        'headache','migraine',
        'body pains','pain relief',
        'asthma','heavy breath',
        'migraine','allergies',
        'cold','sneezing',
        'HighBP','BP'
    ],
    'tablets': [
        'Dextromethorphan tablet(25mg)','Ascoril Expectorant Tablets',
        'Cetirizine (10mg)','Levocetirizine (5mg)',
        'Crocin 650mg','Dolo 650mg',
        'Crocin', 'Dolo',
        'Ibuprofen (200mg–400mg)','Diclofenac (50mg)',
        'Salbutamol (100mcg)','Levosalbutamol (50mcg)',
        'Sumatriptan (50mg)','Rizatriptan (10mg)',
        'Cheston Cold Tablet (20mg)','Cetzine Cold Tablet (30mg)',
        'lisinopril','enalapril (Vasotec)'
    ]
}
df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df['disease'], df['tablets'], test_size=0.25, random_state=42)


model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

import pickle

with open("disease_tablet.pkl", "wb") as f:
    pickle.dump(model, f)

