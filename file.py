import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained model and vectorizer using joblib
model = joblib.load("lr_model.jb")
vectorizer = joblib.load("vectorizer.jb")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Streamlit UI
st.title("📰 News Detector: Fake or Real?")

st.write("Enter a news article below to check whether it's **Fake** or **Real**.")

# Input field for news text
news_text = st.text_area("Enter News Content Here:", height=200)

if st.button("Check News"):
    if news_text.strip():  # Ensure input is not empty
        # Preprocess and transform text
        processed_text = preprocess_text(news_text)
        vectorized_text = vectorizer.transform([processed_text])

        # Predict
        prediction = model.predict(vectorized_text)[0]
        result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"

        # Display result
        st.subheader("Prediction Result:")
        st.markdown(f"**{result}**")
    else:
        st.warning("⚠️ Please enter some news text to analyze.")

# Footer
st.markdown("---")
st.caption("Developed By HaseebCheema❤️ using Streamlit")
