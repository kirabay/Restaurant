import pandas as pd
df = pd.read_csv('Restaurant_Reviews.tsv',delimiter="\t",quoting=3)
import re
import nltk
# import nltk
# nltk.download('all')

# nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(0,1000):   #we have 1000 reviews
    review = re.sub('[^a-zA-Z]'," ",df["Review"][i])
    review = review.lower()
    review = review.split()
    all_stopword = stopwords.words('english')
    all_stopword.remove('not')
    review = [wordnet_lemmatizer.lemmatize(word) for word in review if not word in set(all_stopword)]
    review = " ".join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
y = df['Liked']
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
cls = MultinomialNB().fit(X_train, y_train)
y_pred=cls.predict(X_test)
import pickle
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))      
pickle.dump(cls, open("review.pkl", "wb"))
import streamlit as st
import pickle
## Heading 
st.write('''
#  Hotel Review Sentiment Analysis
''')
## Subheading
st.write("A Web app that detects whether an Review is Positive or Negative")
## Text Input Field
review = st.text_input("Enter Your Review...")
## Button For Generating Predictions
Generate_pred = st.button("Predict Sentiment")
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
## Loading Vectorizer
cv = pickle.load(open('vectorizer.pkl','rb'))
## Function for cleaning text and transforming into vectors
def clean_review(review):
  new_review = re.sub('[^a-zA-Z]', ' ', review)
  new_review = new_review.lower()
  new_review = new_review.split()
  
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  new_review = [ps.stem(word) for word in new_review 
                           if not word in set(all_stopwords)]
  new_review = ' '.join(new_review)
  new_corpus = [new_review]
  new_X_test = cv.transform(new_corpus).toarray()
  return new_X_test
new_X_test = clean_review(review)
## Loading ML Model
model = pickle.load(open('review.pkl','rb'))
## Generating prediction on button click
if Generate_pred:
  pred = model.predict(new_X_test)
  if review!="":
    if pred==1:
      st.write("Positive 😀")
    else:
      st.write("Negative 😑")
