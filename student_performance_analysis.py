import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("student-mat.csv", sep=";")

# -----------------------
# ENCODE ONLY CATEGORICAL COLUMNS (FIXED)
# -----------------------
df_encoded = df.copy()

for col in df.columns:
    if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
        df_encoded[col] = LabelEncoder().fit_transform(df[col])

df = df_encoded

# -----------------------
# REGRESSION (Predict G3)
# -----------------------
X_reg = df.drop("G3", axis=1)
y_reg = df["G3"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Define and train model
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)

# Predict
y_pred_lr = lr.predict(X_test_reg)

print("=== REGRESSION RESULTS ===")
print("MSE:", mean_squared_error(y_test_reg, y_pred_lr))
print("R2:", r2_score(y_test_reg, y_pred_lr))

# -----------------------
# CLASSIFICATION (Pass/Fail)
# -----------------------
df["pass"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

X_cls = df.drop(["G3", "pass"], axis=1)
y_cls = df["pass"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC()
}

print("\n=== CLASSIFICATION RESULTS ===")

for name, model in models.items():
    model.fit(X_train_cls, y_train_cls)
    y_pred = model.predict(X_test_cls)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test_cls, y_pred))
    print("Precision:", precision_score(y_test_cls, y_pred))
    print("Recall:", recall_score(y_test_cls, y_pred))
    print("F1 Score:", f1_score(y_test_cls, y_pred))

# -----------------------
# CONFUSION MATRIX (Random Forest)
# -----------------------
rf = models["Random Forest"]
y_pred_rf = rf.predict(X_test_cls)

cm = confusion_matrix(y_test_cls, y_pred_rf)

print("\nConfusion Matrix (Random Forest):")
print(cm)

# -----------------------
# FEATURE IMPORTANCE GRAPH
# -----------------------
importances = rf.feature_importances_
features = X_cls.columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# -----------------------
# FUNCTION: SAVE TABLE AS IMAGE
# -----------------------
def save_table_as_image(df, title, filename):
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# -----------------------
# TABLE 1: Regression
# -----------------------
table1 = pd.DataFrame({
    "Model": ["Linear Regression"],
    "MSE": [mean_squared_error(y_test_reg, y_pred_lr)],
    "R2": [r2_score(y_test_reg, y_pred_lr)]
})

save_table_as_image(table1, "Table 1: Regression Results", "table1.png")

# -----------------------
# TABLE 2: Classification
# -----------------------
table2 = pd.DataFrame({
    "Model": list(models.keys()),
    "Accuracy": [accuracy_score(y_test_cls, models[m].predict(X_test_cls)) for m in models],
    "Precision": [precision_score(y_test_cls, models[m].predict(X_test_cls)) for m in models],
    "Recall": [recall_score(y_test_cls, models[m].predict(X_test_cls)) for m in models],
    "F1 Score": [f1_score(y_test_cls, models[m].predict(X_test_cls)) for m in models]
})

save_table_as_image(table2, "Table 2: Classification Results", "table2.png")

# -----------------------
# TABLE 3: Confusion Matrix
# -----------------------
cm_df = pd.DataFrame(cm,
                     index=["Actual Pass", "Actual Fail"],
                     columns=["Predicted Pass", "Predicted Fail"])

save_table_as_image(cm_df, "Table 3: Confusion Matrix", "table3.png")

# -----------------------
# ACCURACY COMPARISON GRAPH
# -----------------------
model_names = list(models.keys())
accuracies = [accuracy_score(y_test_cls, models[m].predict(X_test_cls)) for m in models]

plt.figure()
plt.bar(model_names, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.close()

print("\nSaved: feature_importance.png, table1.png, table2.png, table3.png, accuracy_comparison.png")
