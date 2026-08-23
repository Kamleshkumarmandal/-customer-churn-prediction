📊 Customer Churn Prediction

A Machine Learning web application that predicts whether a telecom customer is likely to churn based on their personal, service, and account information.

The application is built using **Python, Pandas, Scikit-learn, Joblib, and Streamlit**.

---

🚀 Project Overview

Customer churn refers to customers leaving a company or stopping its services.

This project uses a **Random Forest Classifier** to predict customer churn based on information such as:

- Gender
- Senior Citizen status
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

The application provides an interactive web interface where users can enter customer information and receive a churn prediction along with the probability of churn.

---

✨ Features

- 📊 Customer churn prediction
- 🤖 Random Forest machine learning model
- 🔄 Automated categorical data preprocessing
- 📈 Churn probability prediction
- 🖥️ Interactive Streamlit web interface
- 💾 Trained model saved using Joblib
- 📁 Organized project structure
- 🎯 User-friendly customer information form

---

🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading and data processing |
| Scikit-learn | Machine learning and preprocessing |
| Random Forest | Classification model |
| Joblib | Saving and loading trained model |
| Streamlit | Interactive web application |
| CSV | Dataset storage |

---

📂 Project Structure

```text
Customer-Churn-App/
│
├── data/
│   └── telco_customer_churn.csv.csv
│
├── app.py
├── train_model.py
├── churn_model.pkl
└── README.md
