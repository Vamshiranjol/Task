import spacy
import joblib
import pickle



# Load the scispaCy model


nlp = spacy.load("en_core_sci_sm")  # en_core is a model for medical language

# Sample doctor's note
text = """
Dr. Samantha Rodriguez, a senior cardiologist at St. Mary’s Hospital,
reviewed the patient's echocardiogram results and noted signs of left ventricular hypertrophy, 
likely secondary to longstanding hypertension. She recommended initiating a beta-blocker in 
combination with ACE inhibitors, advised dietary sodium restriction, and scheduled a cardiology
follow-up in four weeks. During the consultation, the doctor also discussed the importance of
medication adherence, regular blood pressure monitoring, and potential referral to a 
nephrologist if renal function continues to decline."
"""

# Process the text
doc = nlp(text)

model=nlp

# Extract and print the named entities
print("Medical Entities Found:\n")
for ent in doc.ents:
    print(f" {ent.text} [{ent.label_}]")

with open('day12.pkl', 'wb') as f:
    pickle.dump(model, f)
    
