import pandas 
import numpy 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report

df = pandas.read_csv("emails.csv")
df = df[['text' , 'spam']]
df = df.dropna(axis=0, how='any')
x_train , x_test , y_train , y_test = train_test_split(
    df['text'] , df['spam'] , test_size=0.2 , random_state=11
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
    prediction = model.predict(text_vec)[0]
    return "Spam" if prediction == 1 else "Not Spam"

email = "Congratulations! You've won a free iPhone. Click here now!"
print("\nTest Email Prediction:", predict_email(email))