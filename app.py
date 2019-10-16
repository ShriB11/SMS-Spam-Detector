from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    
    #Lower casing
    df['message'] = df['message'].str.lower()
    #Nltk for the first time. Run this
    #nltk.download()
    
    #Tokenize the words
    lines = df.message
    tokenize = []
    for word in lines:
        tokenize.append(word_tokenize(word))
    df.insert(1,"tokenized",lines)
    
    #removed '.' and ',' completely
    clean = [[word for word in lines if word != ',' if word !='.'] for lines in df['tokenized']]
    df.insert(2,"clean",clean)
    
    #Stem the words
    porter_stemmer = PorterStemmer()
    #Stem the words
    stemmed_words = [[porter_stemmer.stem(word = word) for word in lines] for lines in df['tokenized']]
    df.insert(3,"stemmed",stemmed_words)
    
    #Lemmatize the words
    #if not installed
    #nltk.download('wordnet') 
    lemmatizer = WordNetLemmatizer()
    lemmatized = [[lemmatizer.lemmatize(word=word,pos='v') for word in lines] for lines in df['stemmed']]
    df.insert(4,"lemmatized",lemmatized)
    
    #stop words removal
    stop_words = set(stopwords.words('english'))
    stop_remove = [[lemmatizer.lemmatize(word=word,pos='v') for word in lines if word not in stop_words] for lines in df['stemmed']]
    df.insert(5,"rem_stop",stop_remove)
    df['rem_stop'] = df['rem_stop'].apply(', '.join)
    X = df['tokenized']
    y = df['label']

    #Modelling
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfTransformer


    #clf = MultinomialNB()
    #clf.fit(X_train, y_train)
    #clf.score(X_test, y_test)

    nb = Pipeline([('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
    nb.fit(X_train, y_train)

    # Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = nb.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
