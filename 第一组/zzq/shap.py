import shap
X,y = shap.datasets.adult()
X_display, y_display = shap.datasets.adult(display=True)

explainer = shap.DeepExplainer(model) 
shap_values = explainer.shap_values(X) 
shap_values2 = explainer(X) 