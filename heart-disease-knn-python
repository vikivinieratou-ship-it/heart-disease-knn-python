# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# 1. Load dataset

heart_data = pd.read_csv("heart.csv")

print("First rows of dataset:")
print(heart_data.head())



# 2. Basic dataset information

print("\nDataset structure:")
print(heart_data.info())

print("\nNumber of records:", len(heart_data))

print("\nType of 'oldpeak':", heart_data['oldpeak'].dtype)

print("\nValue of 'thalach' for 18th record:", heart_data.loc[17, 'thalach'])



# 3. Check for missing values

missing_values = heart_data.isna().sum().sum()
print("\nNumber of missing values:", missing_values)


# 4. Normalize numeric features

scaler = MinMaxScaler()

numeric_features = ['age','trestbps','chol','thalach','oldpeak']

heart_data[numeric_features] = scaler.fit_transform(
    heart_data[numeric_features]
)


# 5. One-hot encoding

categorical_cols = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]

heart_data = pd.get_dummies(heart_data, columns=categorical_cols)


# 6. Train-test split (80-20)

X = heart_data.drop("target", axis=1)
y = heart_data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)



# 7. kNN Model

k = 5

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

print("\nFirst predictions:")
print(predictions[:10])


# 8. Model Evaluation

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)

print("\nAccuracy:", round(accuracy,3))
print("Precision:", round(precision,3))
print("Recall:", round(recall,3))

print("\nConfusion Matrix:")
print(cm)


# 9. Hyperparameter tuning

k_values = range(1,21,2)
accuracy_scores = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    accuracy_scores.append(acc)


results = pd.DataFrame({
    "k": k_values,
    "Accuracy": accuracy_scores
})

print("\nAccuracy for different k values:")
print(results)



# 10. Plot accuracy vs k

plt.figure(figsize=(8,5))

plt.plot(results["k"], results["Accuracy"], marker="o")

plt.title("Accuracy vs k in kNN")
plt.xlabel("k value")
plt.ylabel("Accuracy")

plt.grid()

plt.show()
