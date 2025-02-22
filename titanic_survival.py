import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = "./Titanic-Dataset.csv"  # Ensure this file is in the same directory
df = pd.read_csv(file_path)

# Drop irrelevant columns
df_cleaned = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Fill missing Age values with the median age
imputer = SimpleImputer(strategy="median")
df_cleaned["Age"] = imputer.fit_transform(df_cleaned[["Age"]])

# Fill missing Embarked values with the most frequent value
df_cleaned["Embarked"].fillna(df_cleaned["Embarked"].mode()[0], inplace=True)

# Encode categorical variables (Sex, Embarked)
label_encoders = {}
for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

# Separate features and target
X = df_cleaned.drop(columns=["Survived"])
y = df_cleaned["Survived"]

# Standardize numerical features
scaler = StandardScaler()
X[["Age", "Fare"]] = scaler.fit_transform(X[["Age", "Fare"]])

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
