import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    st.title("Tennis Data Analysis")

    # Load data from CSV
    data = pd.read_csv('tennisdata.csv')
    st.subheader("First 5 rows of data:")
    st.write(data.head())

    # Obtain Train data and Train output
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Convert to numbers
    le_outlook = LabelEncoder()
    X.Outlook = le_outlook.fit_transform(X.Outlook)

    le_Temperature = LabelEncoder()
    X.Temperature = le_Temperature.fit_transform(X.Temperature)

    le_Humidity = LabelEncoder()
    X.Humidity = le_Humidity.fit_transform(X.Humidity)

    le_Windy = LabelEncoder()
    X.Windy = le_Windy.fit_transform(X.Windy)

    le_PlayTennis = LabelEncoder()
    y = le_PlayTennis.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Train the classifier
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)

    st.subheader("Model Evaluation:")
    st.write("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
