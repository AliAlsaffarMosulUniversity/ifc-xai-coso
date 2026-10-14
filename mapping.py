import pandas as pd
d = pd.read_csv("table4_shap_feature_importance.csv").head(15).copy()
m = {
 'cat__status_no checking account':'Risk Assessment','cat__savings_... < 100 DM':'Risk Assessment',
 'cat__credit_history_critical account/other credits existing':'Risk Assessment','num__installment_rate':'Control Activities',
 'num__duration':'Control Activities','cat__status_... < 100 DM':'Risk Assessment','num__number_credits':'Control Activities',
 'cat__personal_status_sex_male : single':'Control Environment','cat__employment_duration_4 <= ... < 7 years':'Control Environment',
 'cat__property_unknown/no property':'Risk Assessment','cat__housing_rent':'Control Environment','num__amount':'Control Activities',
 'cat__savings_unknown/no savings account':'Risk Assessment','cat__status_0 <= ... < 200 DM':'Risk Assessment','cat__property_real estate':'Risk Assessment'}
d["COSO_Component"] = d["Feature"].map(m)
d.to_csv("table5_coso_mapping_detail.csv", index=False)
a = d.groupby("COSO_Component")["Mean|SHAP value|"].agg(["count","sum"]).reset_index()
a["share_%"] = (a["sum"]/a["sum"].sum()*100).round(1)
a = a.sort_values("sum", ascending=False)
a.to_csv("table6_coso_component_share.csv", index=False)
print(a.to_string(index=False))
with pd.ExcelWriter("results_tables.xlsx") as w:
    pd.read_csv("table1_descriptive_stats.csv", index_col=0).to_excel(w, sheet_name="1_Descriptive")
    pd.read_csv("table3_model_performance.csv").to_excel(w, sheet_name="3_Model_Performance", index=False)
    pd.read_csv("table4_shap_feature_importance.csv").to_excel(w, sheet_name="4_SHAP_All_Features", index=False)
    d.to_excel(w, sheet_name="5_COSO_Mapping_Top15", index=False)
    a.to_excel(w, sheet_name="6_COSO_Share", index=False)
