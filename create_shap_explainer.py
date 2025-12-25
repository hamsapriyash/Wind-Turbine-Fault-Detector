# create_shap_explainer.py
import joblib
import shap

MODEL_PATH = "final_model.joblib"
EXPLAINER_PATH = "shap_explainer.joblib"

# 1. Load your trained model
model = joblib.load(MODEL_PATH)
print("Loaded model from", MODEL_PATH)

# 2. Create SHAP TreeExplainer for this model
explainer = shap.TreeExplainer(model)
print("SHAP TreeExplainer created.")

# 3. Save explainer as joblib
joblib.dump(explainer, EXPLAINER_PATH)
print("Saved SHAP explainer to", EXPLAINER_PATH)
