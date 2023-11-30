from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import spacy

app = Flask(__name__)

# Load and preprocess data
columns = ['id','country','Label','Text']
df = pd.read_csv("twitter_training.csv", names=columns, nrows=10000)
df.dropna(inplace=True)

nlp = spacy.load("en_core_web_sm") 

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens) 

df['Preprocessed Text'] = df['Text'].apply(preprocess)

# Encoding target column
le_model = LabelEncoder()
df['Label'] = le_model.fit_transform(df['Label'])

# Create and Train Naive Bayes Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Preprocessed Text'])
y = df['Label']
nb_clf = MultinomialNB()
nb_clf.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classification_report')
def classification_report():
    return render_template('classification_report.html')

@app.route('/class_distribution')
def class_distribution():
    return render_template('class_distribution.html')

@app.route('/heatmap')
def heatmap():
    return render_template('heatmap.html')

@app.route('/feature_identity')
def feature_identity():
    return render_template('feature_identity.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    user_input_processed = [preprocess(user_input)]
    user_input_vectorized = vectorizer.transform(user_input_processed)
    prediction = le_model.inverse_transform(nb_clf.predict(user_input_vectorized))[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
