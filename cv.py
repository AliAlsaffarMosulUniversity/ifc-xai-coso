import pandas as pd, numpy as np, json
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, recall_score, f1_score, accuracy_score, precision_score, confusion_matrix
df = pd.read_csv("GermanCredit.csv"); df["risk"]=1-df["credit_risk"]; df=df.drop(columns=["credit_risk"])
y=df["risk"]; X=df.drop(columns=["risk"])
cat=X.select_dtypes(include=["object","string"]).columns.tolist(); num=[c for c in X.columns if c not in cat]
pre=ColumnTransformer([("n",StandardScaler(),num),("c",OneHotEncoder(handle_unknown="ignore"),cat)])
M={"Logistic Regression":LogisticRegression(max_iter=2000,random_state=42),
"Decision Tree":DecisionTreeClassifier(max_depth=6,random_state=42),
"Random Forest":RandomForestClassifier(n_estimators=300,max_depth=8,random_state=42),
"Gradient Boosting":GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=42)}
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42)
res={k:{"auc":[],"rec":[],"f1":[],"acc":[]} for k in M}
for tr,te in cv.split(X,y):
    for k,c in M.items():
        p=Pipeline([("p",pre),("c",c)]).fit(X.iloc[tr],y.iloc[tr])
        pr=p.predict_proba(X.iloc[te])[:,1]; yp=(pr>=.5).astype(int)
        res[k]["auc"].append(roc_auc_score(y.iloc[te],pr)); res[k]["rec"].append(recall_score(y.iloc[te],yp))
        res[k]["f1"].append(f1_score(y.iloc[te],yp)); res[k]["acc"].append(accuracy_score(y.iloc[te],yp))
rows=[]
for k,v in res.items():
    rows.append([k]+[f"{np.mean(v[m]):.4f}" for m in ["acc","rec","f1","auc"]]+[f"{np.std(v['auc'],ddof=1):.4f}"])
pd.DataFrame(rows,columns=["Model","Accuracy","Recall","F1-Score","AUC-ROC (mean)","AUC SD"]).to_csv("table7_cv_results.csv",index=False)
t,p=stats.ttest_rel(res["Logistic Regression"]["auc"],res["Random Forest"]["auc"])
d=np.array(res["Logistic Regression"]["auc"])-np.array(res["Random Forest"]["auc"])
# bootstrap CI for LR test AUC (same split as main)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
lr=Pipeline([("p",pre),("c",M["Logistic Regression"])]).fit(Xtr,ytr); pr=lr.predict_proba(Xte)[:,1]
rng=np.random.default_rng(42); b=[]
yt=yte.values
for _ in range(2000):
    i=rng.integers(0,len(yt),len(yt))
    if len(set(yt[i]))<2: continue
    b.append(roc_auc_score(yt[i],pr[i]))
ci=(np.percentile(b,2.5),np.percentile(b,97.5))
# threshold analysis
th=[]
for t_ in [0.5,0.4,0.3,0.25,0.2]:
    yp=(pr>=t_).astype(int); cm=confusion_matrix(yt,yp)
    th.append([t_,round(accuracy_score(yt,yp),4),round(precision_score(yt,yp),4),round(recall_score(yt,yp),4),round(f1_score(yt,yp),4),int(cm[1,0]),int(cm[0,1])])
pd.DataFrame(th,columns=["Threshold","Accuracy","Precision","Recall","F1","FN","FP"]).to_csv("table8_threshold.csv",index=False)
json.dump({"paired_t":float(t),"paired_p":float(p),"mean_diff":float(d.mean()),"ci_lo":float(ci[0]),"ci_hi":float(ci[1]),"folds":len(d)},open("cv_summary.json","w"),indent=2)
print(pd.read_csv("table7_cv_results.csv").to_string(index=False)); print(pd.read_csv("table8_threshold.csv").to_string(index=False)); print(open("cv_summary.json").read())
