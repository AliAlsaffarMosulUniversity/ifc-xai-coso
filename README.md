# IFC-XAI: Explainable AI Governance in Internal Financial Control (COSO-aligned)

Code and results accompanying the paper:
**"Governance of Explainable AI in Internal Financial Control Systems: A Proposed Framework Integrating XAI Techniques with the COSO Model to Enhance Institutional Sustainability"**

Author: Ali Abdulwahab Yahya Al-Saffar — College of Administration and Economics, University of Mosul.

## What this repo does
- Trains four classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) on the public German Credit dataset (1000 records).
- Explains Logistic Regression with SHAP and maps the top-15 features to COSO components.
- Robustness: repeated stratified 5-fold CV (x3), bootstrap CI for AUC, decision-threshold analysis.
- Draws the four-layer IFC-XAI conceptual framework.

## Run
```bash
pip install -r requirements.txt
python analysis.py
python mapping.py
python cv.py
python draw_fig4.py
```
Random seed is 42 everywhere, so the numbers match the paper.

## Key results (test set, 300 cases)
| Model | AUC-ROC |
|---|---|
| Logistic Regression | 0.8011 |
| Random Forest | 0.7971 |
| Gradient Boosting | 0.7651 |
| Decision Tree | 0.6326 |

Under repeated CV the Logistic Regression vs Random Forest AUC difference is not statistically significant (paired t-test, p = 0.058).

## Data
German Credit (Statlog), UCI Machine Learning Repository — Hofmann (1994); Dua & Graff (2019).

## Notes
- `REFERENCES_CHECK.md` lists which references were verified.
- Code is released under the MIT License (see LICENSE). The German Credit dataset remains subject to its UCI terms.

## Citation / Attribution
This work is free to use. Please credit the author (see `LICENSE` and `CITATION.cff`):

Al-Saffar, A. A. Y. (2026). *IFC-XAI: Explainable AI Governance in Internal Financial Control (COSO-aligned)* [Software]. https://github.com/AliAlsaffarMosulUniversity/ifc-xai-coso
