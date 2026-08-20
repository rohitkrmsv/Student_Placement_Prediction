import pandas as pd

data = pd.read_csv("data/students.csv")

print("First 5 students:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nDataset information:")
print(data.info())

print("\nBasic statistics:")
print(data.describe())
X = data.drop("Placed", axis=1)
y = data["Placed"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)

print("\nTraining target:")
print(y_train.shape)

print("\nTesting target:")
print(y_test.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel training completed!")
y_pred = model.predict(X_test)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))