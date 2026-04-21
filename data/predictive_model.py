import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

sns.set(style="whitegrid")

os.makedirs("data", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

data = {
    "age":[25,30,45,35,22,40,28,32,38,27,29,41,36,33,26,39,31,34,37,24],
    "salary":[50000,60000,80000,75000,48000,90000,52000,61000,85000,57000,62000,95000,78000,69000,54000,88000,63000,72000,81000,50000],
    "experience":[2,5,10,7,1,12,3,6,9,4,5,13,8,6,3,11,5,7,9,2],
    "department":["Sales","HR","IT","Finance","Sales","IT","HR","Finance","IT","Sales","HR","IT","Finance","Sales","HR","IT","Finance","Sales","IT","HR"],
    "target":[0,1,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,1,0]
}

df = pd.DataFrame(data)

le = LabelEncoder()
df["department"] = le.fit_transform(df["department"])

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, preds)
    results[name] = acc

    cm = confusion_matrix(y_test, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"visuals/{name}_cm.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend()
    plt.savefig(f"visuals/{name}_roc.png")
    plt.close()

    print(name)
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

best_model = max(results, key=results.get)
print("Best Model:", best_model)
