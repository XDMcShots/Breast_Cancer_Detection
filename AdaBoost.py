import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Please ignore this typo, we have just accepted it at this point
df = pd.read_csv("Chronic_Kidney_Dsease_data_Classification.csv")

X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

y = y.astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

adaboost_clf = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

adaboost_clf.fit(X_train, y_train)

y_pred = adaboost_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost Classifier Accuracy: {accuracy:.4f}")
