import pandas 
import numpy 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report

df = pandas.read_csv("spam.csv")
df = df[['text' , 'spam']]
df['spam'] = df['spam'].map({'ham': 0, 'spam': 1})
df = df.dropna()
x_train , x_test , y_train , y_test = train_test_split(
    df['text'] , df['spam'] , test_size=0.2 , random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english' , max_df=0.7)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)


model = MultinomialNB()
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)

print("accuracy : " , accuracy_score(y_test , y_pred))
print("\nClassification Report : \n" , classification_report(y_test , y_pred))



def predict_email(text):
    text_vec = vectorizer.transform([text])
    prob = model.predict_proba(text_vec)[0][1]
    return f"Spam ({prob:.2f})" if prob > 0.5 else f"Not Spam ({prob:.2f})"

email = "You are a winner, you are won free iphone"
print("\nTest Email Prediction:", predict_email(email))