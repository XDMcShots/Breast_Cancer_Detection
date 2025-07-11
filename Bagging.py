import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("Chronic_Kidney_Dsease_data_Classification.csv")   # replace with your CSV file name

X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base_estimator = DecisionTreeClassifier(random_state=42)

bagging_clf = BaggingClassifier(
    estimator=base_estimator,
    n_estimators=50,
    max_samples=0.8,
    max_features=1.0,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# 🚀 Train the model
bagging_clf.fit(X_train, y_train)

# 🔮 Make predictions
y_pred = bagging_clf.predict(X_test)

# 📈 Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Bagging Classifier Accuracy: {accuracy:.4f}")
