# 🛒 Customer Purchase Prediction – Streamlit App

An **interactive Machine Learning web application** built with **Python, Scikit-learn, and Streamlit** that predicts whether a customer is likely to make a purchase based on the number of visits.

This project focuses on **visualizing ML predictions** in a clean, user-friendly interface suitable for demos, portfolios, and client presentations.

---

## 🚀 Features

- Real-time purchase prediction
- Confidence score display
- Clean and professional UI
- Trained ML model integration
- Simple user input (number of visits)
- Deployment-ready structure

---

## 🧠 Machine Learning Model

- Algorithm: Logistic Regression
- Feature: `customer_visits`
- Target: `purchase` (1 = Yes, 0 = No)

The model is pre-trained and loaded using `joblib`.

---

## 📂 Project Structure

# Streamlit web app
streamlit_app.py 

# Trained ML model
customer_purchase_model.pkl 

# Dependencies
requirements.txt



---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/zeeshandev6066/customer-purchase-streamlit-app.git
cd customer-purchase-streamlit-app

Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run streamlit_app.py


The app will open automatically in your browser.


How It Works

User enters the number of customer visits

Model predicts purchase behavior

App displays:

Purchase decision

Confidence percentage

🛠️ Technologies Used

Python
Streamlit
Scikit-learn
Joblib
NumPy

#Use Cases

Customer behavior analysis
Marketing prediction demos
Machine learning portfolio project
Freelancing project showcase

🌍 Deployment Ready

This app can be deployed on:

Streamlit Cloud
Hugging Face Spaces
Render

Just upload the repository and connect it to a hosting platform.

#Future Improvements

Multiple input features
API integration
Model comparison
User authentication

#Author

Muhammad Zeeshan
Machine Learning & Python Developer


⭐ Support

If you find this project useful, give it a ⭐ and feel free to fork!
