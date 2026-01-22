import pickle
import gradio as gr
import pandas as pd


# --- 1. Class Definition (REQUIRED for pickle loading) ---
class HybridTacticalModel:
    def __init__(self, preprocessor, prob_model, anomaly_model):
        self.preprocessor = preprocessor
        self.prob_model = prob_model
        self.anomaly_model = anomaly_model

    def analyze(self, X_raw):
        X_proc = self.preprocessor.transform(X_raw)
        risk_prob = self.prob_model.predict_proba(X_proc)[0][1]
        is_anomaly = self.anomaly_model.predict(X_proc)[0] == -1
        return {"probability": float(risk_prob), "anomaly_detected": bool(is_anomaly)}


# --- 2. Load the Model ---
with open("model.pkl", "rb") as f:
    hybrid_system = pickle.load(f)

# --- 3. UI Constants ---
SECTORS = {
    "Teknaf Border": (20.86, 92.30),
    "Ukhiya Zone": (21.16, 92.14),
    "Bandarban Hills": (22.19, 92.21),
    "Sylhet Border": (24.89, 91.86),
    "Dhaka Central": (23.81, 90.41),
    "Chittagong Port": (22.34, 91.83),
    "Cox's Bazar Coast": (21.42, 91.98),
    "Rangamati Hills": (22.65, 92.19),
    "Khulna Sundarbans": (22.45, 89.54),
    "Rajshahi Border": (24.37, 88.60),
    "Comilla Border": (23.47, 91.18),
    "Dinajpur Border": (25.63, 88.67),
}

MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

V_TYPES = {"State-based Action": 1, "Non-state Activity": 2, "Civilian Attacks": 3}


# --- 4. Prediction Function ---
def hybrid_predict(year, sector, month, violence):
    try:
        # A. Map Inputs
        lat, lon = SECTORS[sector]
        m_num = MONTHS[month]
        vt = V_TYPES[violence]

        # B. Create input DataFrame
        input_data = pd.DataFrame(
            [[float(lat), float(lon), int(year), int(m_num), int(vt)]],
            columns=["latitude", "longitude", "year", "month", "type_of_violence"],
        )

        # C. Get Hybrid Intelligence
        analysis = hybrid_system.analyze(input_data)
        risk_prob = analysis["probability"]
        anomaly = analysis["anomaly_detected"]

        # D. Unified Status Logic
        if risk_prob >= 0.50:
            status = "🔴 RED ALERT: High Probability Border Threat"
        elif anomaly or risk_prob >= 0.25:
            status = "🟡 YELLOW ALERT: Unusual Activity (Anomaly Detected)"
        else:
            status = "🟢 GREEN: Routine Internal Patterns"

        return {
            "Tactical_Status": status,
            "Intelligence_Signals": {
                "Historical_Pattern_Match_Prob": f"{risk_prob:.2%}",
                "Outlier_Detection_Warning": "ACTIVE" if anomaly else "None",
            },
            "Commander_Note": "Yellow alerts indicate events that don't match standard internal conflict patterns and should be verified by ground sensors/scouts.",
            "Input_Context": {
                "Year": int(year),
                "Sector": sector,
                "Month": month,
                "Conflict_Type": violence,
            },
        }

    except Exception as e:
        return {"Error": str(e)}


# --- 5. Interface Construction ---
ui = gr.Interface(
    fn=hybrid_predict,
    inputs=[
        gr.Number(value=2026, label="Forecast Year"),
        gr.Dropdown(
            choices=list(SECTORS.keys()),
            label="Sector Selection",
            value="Teknaf Border",
        ),
        gr.Dropdown(choices=list(MONTHS.keys()), label="Forecast Month", value="Jan"),
        gr.Dropdown(
            choices=list(V_TYPES.keys()),
            label="Conflict Category",
            value="State-based Action",
        ),
    ],
    outputs=gr.JSON(label="Tactical Intelligence Report"),
    title="Hybrid Early Warning System (HEWS)",
    description="A theoretical dual-signal intelligence system combining Historical Probability (AdaBoost) and Anomaly Detection (SVM). Designed for cross-border conflict prediction. Select parameters and receive a tactical status report. Yellow alerts indicate events that don't match standard internal conflict patterns and should be verified by ground sensors/scouts. The dataset used is from UCDP Border Crossing dataset exported on 17th Januay 2026, it only includes conflict data up to year 2024.",
)

# --- 6. Launch ---
if __name__ == "__main__":
    ui.launch(
        share=True,
        pwa=True
    )
    