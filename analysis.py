import pandas as pd, numpy as np, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.read_csv("GermanCredit.csv")
df["risk"] = 1 - df["credit_risk"]
df = df.drop(columns=["credit_risk"])
y = df["risk"]
X = df.drop(columns=["risk"])
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

desc = X[num_cols].describe().T
desc["missing"] = X[num_cols].isna().sum()
desc.to_csv("table1_descriptive_stats.csv")

pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
}
results, roc_data, fitted = [], {}, {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    yp = pipe.predict(X_test)
    ypr = pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, yp)
    fpr, tpr, _ = roc_curve(y_test, ypr)
    auc = roc_auc_score(y_test, ypr)
    roc_data[name] = (fpr, tpr, auc)
    results.append({"Model": name, "Accuracy": round(accuracy_score(y_test, yp), 4),
                    "Precision": round(precision_score(y_test, yp), 4),
                    "Recall": round(recall_score(y_test, yp), 4),
                    "F1-Score": round(f1_score(y_test, yp), 4), "AUC-ROC": round(auc, 4),
                    "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]})
res_df = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False)
res_df.to_csv("table3_model_performance.csv", index=False)
print(res_df.to_string(index=False))

plt.figure(figsize=(6, 5))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Financial Risk Prediction Models")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("fig1_roc_curves.png", dpi=150); plt.close()

best_name = res_df.iloc[0]["Model"]
best_pipe = fitted[best_name]
cm_best = confusion_matrix(y_test, best_pipe.predict(X_test))
plt.figure(figsize=(5.2, 5))
plt.imshow(cm_best, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm_best[i, j]), ha="center", va="center", fontsize=14)
plt.xticks([0, 1], ["Pred: No Risk", "Pred: Risk"])
plt.yticks([0, 1], ["Actual: No Risk", "Actual: Risk"])
plt.title(f"Confusion Matrix - {best_name}", fontsize=13)
plt.tight_layout()
plt.savefig("fig2_confusion_matrix.png", dpi=150, bbox_inches="tight", pad_inches=0.3); plt.close()

Xtr = best_pipe.named_steps["pre"].transform(X_train)
Xte = best_pipe.named_steps["pre"].transform(X_test)
fn = best_pipe.named_steps["pre"].get_feature_names_out()
clf_best = best_pipe.named_steps["clf"]
if best_name == "Logistic Regression":
    sv = shap.LinearExplainer(clf_best, Xtr).shap_values(Xte)
else:
    sv = shap.TreeExplainer(clf_best).shap_values(Xte)
    if isinstance(sv, list): sv = sv[1]
    elif sv.ndim == 3: sv = sv[:, :, 1]

imp = pd.DataFrame({"Feature": fn, "Mean|SHAP value|": np.abs(sv).mean(axis=0)}) \
        .sort_values("Mean|SHAP value|", ascending=False).reset_index(drop=True)
imp["Rank"] = imp.index + 1
imp.to_csv("table4_shap_feature_importance.csv", index=False)
print(imp.head(15).to_string(index=False))

plt.figure(figsize=(11, 8.5))
shap.summary_plot(sv, Xte, feature_names=fn, show=False, max_display=12)
plt.tight_layout()
plt.savefig("fig3_shap_summary.png", dpi=150, bbox_inches="tight", pad_inches=0.5); plt.close()

json.dump({"best_model": best_name, "n_samples": int(len(df)), "n_features_raw": int(X.shape[1]),
           "n_features_encoded": int(len(fn)), "train_size": int(len(X_train)), "test_size": int(len(X_test))},
          open("summary.json", "w"), indent=2)
print("BEST", best_name)
