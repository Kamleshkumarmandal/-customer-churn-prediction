
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# LOAD DATASET
# -----------------------------

data = pd.read_csv("data/telco_customer_churn.csv.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)


# -----------------------------
# CLEAN DATA
# -----------------------------

# TotalCharges mein kuch values blank ho sakti hain
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

data = data.dropna()


# -----------------------------
# TARGET
# -----------------------------

data["Churn"] = data["Churn"].map({
    "Yes": 1,
    "No": 0
})


# CustomerID prediction ke liye useful nahi hai
data = data.drop(columns=["customerID"])


# -----------------------------
# FEATURES & TARGET
# -----------------------------

X = data.drop(columns=["Churn"])
y = data["Churn"]


# -----------------------------
# CATEGORICAL / NUMERICAL
# -----------------------------

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# -----------------------------
# PREPROCESSING
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        ),
        (
            "numeric",
            "passthrough",
            numeric_columns
        )
    ]
)


# -----------------------------
# MODEL
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# -----------------------------
# PIPELINE
# -----------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# TRAIN MODEL
# -----------------------------

pipeline.fit(X_train, y_train)


# -----------------------------
# ACCURACY
# -----------------------------

accuracy = pipeline.score(X_test, y_test)

print("Model trained successfully!")
print("Accuracy:", accuracy)


# -----------------------------
# SAVE MODEL
# -----------------------------

joblib.dump(
    pipeline,
    "churn_model.pkl"
)

print("Model saved as churn_model.pkl")