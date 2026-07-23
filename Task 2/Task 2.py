import os

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# always work from the folder this script is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Work out the scores for one model and print them
def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    fraud_probs = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions)
    scores = {
        "model": name,
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, fraud_probs),
        "avg_precision": average_precision_score(y_test, fraud_probs),
        "true_negatives": matrix[0, 0],
        "false_positives": matrix[0, 1],
        "false_negatives": matrix[1, 0],
        "true_positives": matrix[1, 1],
    }

    print()
    print(name)
    print("Precision:", round(scores["precision"], 4))
    print("Recall:", round(scores["recall"], 4))
    print("F1 score:", round(scores["f1"], 4))
    print("ROC-AUC:", round(scores["roc_auc"], 4))
    print("Average precision (PR-AUC):", round(scores["avg_precision"], 4))
    print("Confusion matrix:")
    print(matrix)
    return scores, fraud_probs


#1 Load the transaction data
transactions = pd.read_csv("creditcard.csv")
os.makedirs("outputs", exist_ok=True)

print("1. Data loaded")
print("Number of rows:", transactions.shape[0])
print("Number of columns:", transactions.shape[1])
print()

#2 Count fraud vs legitimate and chart it
print("2. Class balance")
legit_count = (transactions["Class"] == 0).sum()
fraud_count = (transactions["Class"] == 1).sum()
fraud_percentage = fraud_count / len(transactions) * 100
print("Legitimate payments (0):", legit_count)
print("Fraud payments (1):", fraud_count)
print("Fraud percentage:", round(fraud_percentage, 3))
print()

plt.figure(figsize=(6, 4))
plt.bar(["Legitimate", "Fraud"], [legit_count, fraud_count], color=["steelblue", "darkorange"])
plt.yscale("log")
plt.title("Class Distribution (log scale)")
plt.ylabel("Count")
plt.text(0, legit_count, f"{legit_count:,}", ha="center", va="bottom")
plt.text(1, fraud_count, f"{fraud_count:,}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png")
plt.close()

#3 Split into train and test sets
print("3. Splitting the data")
X = transactions.drop(columns="Class")
y = transactions["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Training rows:", X_train.shape[0])
print("Test rows:", X_test.shape[0])
print()

#4 Build the two SMOTE pipelines and their grid searches
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
rf_pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
])

searches = {
    "Logistic Regression": GridSearchCV(
        lr_pipeline,
        {"smote__k_neighbors": [3, 5, 7], "classifier__C": [0.01, 0.1, 1.0]},
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    ),
    "Random Forest": GridSearchCV(
        rf_pipeline,
        {"smote__k_neighbors": [3, 5, 7], "classifier__max_depth": [10, 20, None]},
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    ),
}

#5 Tune both models with cross validation
print("5. Tuning the models")
print("Tuning metric used by GridSearchCV: roc_auc")
best_settings_rows = []
for name, search in searches.items():
    search.fit(X_train, y_train)
    print(name, "best settings:", search.best_params_)
    best_settings_rows.append({
        "model": name,
        "best_params": str(search.best_params_),
        "best_cv_roc_auc": round(search.best_score_, 4),
    })
pd.DataFrame(best_settings_rows).to_csv("outputs/best_parameters.csv", index=False)
print()

#6 Score both models on the test set
print("6. Test set results")
score_rows = []
model_probs = {}
for name, search in searches.items():
    scores, fraud_probs = evaluate_model(name, search.best_estimator_, X_test, y_test)
    score_rows.append(scores)
    model_probs[name] = fraud_probs
print()

#7 Save the charts and the score table
plt.figure(figsize=(7, 6))
for scores in score_rows:
    name = scores["model"]
    fpr, tpr, _ = roc_curve(y_test, model_probs[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC={scores['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curves.png")
plt.close()

# Precision-recall curve: the fairer view when only 0.173% of rows are fraud
plt.figure(figsize=(7, 6))
for scores in score_rows:
    name = scores["model"]
    precisions, recalls, _ = precision_recall_curve(y_test, model_probs[name])
    plt.plot(recalls, precisions, label=f"{name} (AP={scores['avg_precision']:.3f})")
plt.axhline(y=y_test.mean(), color="k", linestyle="--", label="Random guessing")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/precision_recall_curves.png")
plt.close()

rf_model = searches["Random Forest"].best_estimator_
importances = pd.Series(rf_model.named_steps["classifier"].feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).to_csv("outputs/feature_importances.csv", header=["importance"])
top_features = importances.nlargest(15).sort_values()
plt.figure(figsize=(8, 6))
plt.barh(top_features.index, top_features.values, color="steelblue")
plt.title("Random Forest - Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# Draw each confusion matrix as a real 2x2 grid instead of raw printed numbers
for scores in score_rows:
    counts = [[scores["true_negatives"], scores["false_positives"]],
              [scores["false_negatives"], scores["true_positives"]]]
    plt.figure(figsize=(5, 4.5))
    plt.imshow(counts, cmap="Blues", norm="log")
    plt.xticks([0, 1], ["Predicted legit", "Predicted fraud"])
    plt.yticks([0, 1], ["Actual legit", "Actual fraud"])
    for row in [0, 1]:
        for column in [0, 1]:
            plt.text(column, row, f"{counts[row][column]:,}", ha="center", va="center")
    plt.title(scores["model"] + " - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_" + scores["model"].lower().replace(" ", "_") + ".png")
    plt.close()

comparison = pd.DataFrame(score_rows)
comparison.to_csv("outputs/model_comparison.csv", index=False)
print("7. Charts and model_comparison.csv saved to the outputs folder")
print(comparison.to_string(index=False))
