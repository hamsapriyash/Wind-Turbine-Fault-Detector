from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# ---- Load trained model, scaler, and SHAP explainer ----
MODEL_PATH = "final_model.joblib"
SCALER_PATH = "scaler.joblib"
EXPLAINER_PATH = "shap_explainer.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# SHAP explainer is optional but recommended
try:
    explainer = joblib.load(EXPLAINER_PATH)
    shap_available = True
except Exception as e:
    print("SHAP explainer not available:", e)
    explainer = None
    shap_available = False

# All V-features used by the model
ALL_FEATURES = [f"V{i}" for i in range(1, 41)]


@app.route("/", methods=["GET", "POST"])
def index():
    # ---- 1. Define Sensor Names Mapping ----
    # You can rename these values to your specific sensor names (e.g., "Wind Speed", "Gearbox Temp")
    FEATURE_MAPPING = {
        "V1": "Wind Speed (m/s)",
        "V2": "Wind Direction (°)",
        "V3": "Power Output (kW)",
        "V4": "Rotor Speed (RPM)",
        "V5": "Generator Speed (RPM)",
        "V6": "Blade Pitch Angle (°)",
        "V7": "Yaw Position (°)",
        "V8": "Ambient Temperature (°C)",
        "V9": "Gearbox Bearing Temp (°C)",
        "V10": "Gearbox Oil Temp (°C)",
        "V11": "Trafo Phase 1 Temp (°C)",
        "V12": "Trafo Phase 2 Temp (°C)",
        "V13": "Trafo Phase 3 Temp (°C)",
        "V14": "Generator Phase 1 Temp (°C)",
        "V15": "Generator Phase 2 Temp (°C)",
        "V16": "Generator Phase 3 Temp (°C)",
        "V17": "Generator Bearing 1 Temp (°C)",
        "V18": "Generator Bearing 2 Temp (°C)",
        "V19": "Grid Active Power (kW)",
        "V20": "Grid Reactive Power (kVar)",
        "V21": "Grid Frequency (Hz)",
        "V22": "Grid Voltage Phase 1 (V)",
        "V23": "Grid Voltage Phase 2 (V)",
        "V24": "Grid Voltage Phase 3 (V)",
        "V25": "Grid Current Phase 1 (A)",
        "V26": "Grid Current Phase 2 (A)",
        "V27": "Grid Current Phase 3 (A)",
        "V28": "Nacelle Temperature (°C)",
        "V29": "Hydraulic Oil Pressure (bar)",
        "V30": "Hydraulic Oil Temp (°C)",
        "V31": "Rotor Bearing Temp (°C)",
        "V32": "Trafo Oil Temp (°C)",
        "V33": "Inverter Cabinet Temp (°C)",
        "V34": "Blade 1 Vibration (mm/s)",
        "V35": "Blade 2 Vibration (mm/s)",
        "V36": "Blade 3 Vibration (mm/s)",
        "V37": "Tower Vibration X (mm/s)",
        "V38": "Tower Vibration Y (mm/s)",
        "V39": "Main Shaft Vibration (mm/s)",
        "V40": "Drive Train Vibration (mm/s)"
    }
    
    prediction = None
    prediction_label = None
    probability = None

    contribution_labels = []
    contribution_values = []

    # For errors (e.g., wrong number of values in pasted string)
    error_message = None

    # Store form values so they persist after submit
    input_values = {f: "" for f in ALL_FEATURES}
    bulk_values_text = ""  # textarea content

    if request.method == "POST":
        # 1) Check if bulk pasted values are provided
        bulk_values_text = request.form.get("bulk_values", "").strip()

        data = {}

        if bulk_values_text:
            # User pasted a CSV row: v1,v2,...,v40
            # Split on commas (and possibly spaces)
            raw_parts = bulk_values_text.replace("\n", " ").split(",")
            parts = [p.strip() for p in raw_parts if p.strip() != ""]

            if len(parts) != len(ALL_FEATURES):
                error_message = (
                    f"Expected {len(ALL_FEATURES)} values in pasted row, "
                    f"but got {len(parts)}."
                )
            else:
                # Map to V1..V40 in order
                for feat, val_str in zip(ALL_FEATURES, parts):
                    try:
                        data[feat] = float(val_str)
                    except ValueError:
                        data[feat] = 0.0
                    # also update individual inputs so they show the same values
                    input_values[feat] = val_str
        else:
            # 2) No bulk string: fall back to individual text boxes
            for feat in ALL_FEATURES:
                val_str = request.form.get(feat, "").strip()
                input_values[feat] = val_str
                try:
                    data[feat] = float(val_str)
                except ValueError:
                    data[feat] = 0.0  # fallback

        # Only predict if no error and we have data
        if not error_message and data:
            # Create single-row DataFrame
            df_instance = pd.DataFrame([data])
            X_raw = df_instance[ALL_FEATURES]

            # Apply same scaler as training
            X_scaled = scaler.transform(X_raw)

            # Predict
            pred = model.predict(X_scaled)[0]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_scaled)[0][1]  # probability of class 1 (fault)
            else:
                prob = None

            prediction = int(pred)
            prediction_label = "Normal (0)" if prediction == 0 else "Abnormal / Fault (1)"
            probability = prob

            # If prediction is abnormal, compute SHAP contributions over ALL 40 V-features
            if prediction == 1 and shap_available:
                shap_values = explainer.shap_values(X_scaled)

                # Case 1: old-style list [class0, class1]
                if isinstance(shap_values, list):
                    # Take class 1 (fault) shap values for first (and only) sample
                    sv = shap_values[1][0]  # shape (n_features,)

                # Case 2: array, usually shape (n_samples, n_features, n_classes)
                else:
                    sv_all = shap_values  # keep original
                    # Expect shape (1, n_features, n_classes) or (1, n_features)
                    if sv_all.ndim == 3:
                        # sv_all[0] -> (n_features, n_classes)
                        # take class 1 across features
                        sv = sv_all[0, :, 1]  # shape (n_features,)
                    elif sv_all.ndim == 2:
                        # sv_all[0] -> (n_features,) for binary with single output
                        sv = sv_all[0]
                    else:
                        raise ValueError(f"Unexpected shap_values shape: {sv_all.shape}")

                # Now sv should be 1D of length len(ALL_FEATURES)
                contrib_series = pd.Series(sv, index=ALL_FEATURES)

                # Sort by absolute contribution
                contrib_series = contrib_series.reindex(
                    contrib_series.abs().sort_values(ascending=False).index
                )

                # Take top 10 for the chart
                contrib_series = contrib_series.head(10)

                # Convert V-names to Friendly Names for the chart
                raw_labels = contrib_series.index.tolist()
                contribution_labels = [FEATURE_MAPPING.get(lbl, lbl) for lbl in raw_labels]
                contribution_values = contrib_series.values.tolist()


    return render_template(
        "index.html",
        all_features=ALL_FEATURES,
        input_values=input_values,
        bulk_values_text=bulk_values_text,
        prediction=prediction,
        prediction_label=prediction_label,
        probability=probability,
        contribution_labels=contribution_labels,
        contribution_values=contribution_values,
        shap_available=shap_available,
        error_message=error_message,
        feature_mapping=FEATURE_MAPPING,
    )


if __name__ == "__main__":
    app.run(debug=True)
