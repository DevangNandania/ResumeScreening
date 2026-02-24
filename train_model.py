import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,accuracy_score
from preprocess import clean_text

df = pd.read_csv('Data/res.csv')
df['cleaned_resume'] = df['Resume'].apply(clean_text)
x = df['cleaned_resume']
y = df['Category']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(1,2))
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))
ac = accuracy_score(y_test,y_pred)
print(ac)
print("Train Accuracy:", model.score(x_train, y_train))
print("Test Accuracy:", model.score(x_test, y_test))

joblib.dump(model, 'Model/model.pkl')
joblib.dump(vectorizer, 'Model/vectorizer.pkl')