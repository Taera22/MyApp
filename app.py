# app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pathlib import Path

# ======================= PAGE & THEME =======================
st.set_page_config(page_title="PPM-ML Cameroon", page_icon="🤰", layout="centered")

st.markdown("""
<style>
.stApp {
  background: radial-gradient(1200px 800px at 0% 0%, #eef8f5 0%, transparent 55%),
              radial-gradient(1200px 800px at 100% 10%, #f9f5ee 0%, transparent 52%);
  font-family: Inter, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}
.card { background:#fff; border-radius:20px; padding:18px; box-shadow:0 10px 28px rgba(0,0,0,.06); border:1px solid #eff2f6; }
h1 { color:#237A57; letter-spacing:.2px; margin-bottom:.2rem; }
.subtle { color:#6b7280; font-size:.94rem; }
.stButton>button { border-radius:12px; padding:.65rem 1.05rem; font-weight:600; }
.badge { display:inline-block; padding:.42rem .68rem; border-radius:999px; font-weight:700; font-size:.95rem; }
.badge-low{ background:#E8F7EE; color:#0F7B3E; border:1px solid #cfeedd; }
.badge-mid{ background:#FFF4E5; color:#8A4D00; border:1px solid #ffe5c2; }
.badge-high{ background:#FDEBEC; color:#AF1E2D; border:1px solid #f6cfd3; }
.panel{ border-radius:16px; padding:16px; border:1px solid #edf0f5; }
.panel-low{ background:#F6FFF9; } .panel-mid{ background:#FFF9F0; } .panel-high{ background:#FFF4F5; }
.prob-row{ display:flex; align-items:center; justify-content:space-between; padding:.35rem .6rem; border-radius:10px; background:#fafafa; margin-bottom:.4rem; border:1px solid #f0f2f5; }
.help{ color:#6b7280; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

# ======================= HEADER (centered logo + title) =======================
with st.container():
    c1, c2, c3 = st.columns([1, 2, 1])  # center column wider
    with c2:
        try:
            st.image(Image.open("Design_sans_titre-removebg-preview.png"), use_container_width=True)
        except Exception as e:
            st.warning(f"Logo not found: {e}")
        st.markdown(
            "<h1 style='text-align:center;margin-top:.25rem'>PPM-ML CAMEROON</h1>"
            "<div class='subtle' style='text-align:center'>Bon Maternal Health • App</div>",
            unsafe_allow_html=True,
        )

st.caption("Model trained with SMOTE + class weights (F1-macro). Inputs use realistic ranges; BP category can be auto-derived; clinical guardrails keep edge cases sensible.")

# ======================= LOAD ARTIFACTS (silent & robust) =======================
@st.cache_resource
def load_artifacts():
    model = joblib.load("preeclampsia_rf_model.pkl")
    le = joblib.load("label_encoder.pkl")

    feature_path = Path("feature_names.json")
    if feature_path.exists():
        with feature_path.open("r", encoding="utf-8") as f:
            feats = json.load(f)
    else:
        # Try model feature names
        feats = []
        try:
            feats = list(getattr(model, "feature_names_in_", []))
        except Exception:
            pass
        # Fallback to training list
        if not feats:
            feats = [
                'gravida','parity','gestational age (weeks)','Age (yrs)','BMI  [kg/m²]',
                'diabetes','History of hypertension (y/n)','Systolic BP','Diastolic BP','HB',
                'fetal weight(kgs)','Protien Uria','Uterine Artery Doppler Resistance Index (RI)',
                'amniotic fluid levels(cm)','History of preeclampsia',
                'Uterine Artery Doppler Pulsatility Index (PI)','Number of babies',
                'BodyTemp','HeartRate','Blood Pressure','BS'
            ]
        # Persist silently so future runs are clean
        try:
            feature_path.write_text(json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return model, le, feats

model, label_encoder, feature_names = load_artifacts()
CLASSES = list(label_encoder.classes_)  # e.g. ['high risk','low risk','mid risk']

BADGE = {
    "low risk" : "<span class='badge badge-low'>🟢 Low Risk</span>",
    "mid risk" : "<span class='badge badge-mid'>🟠 Moderate Risk</span>",
    "high risk": "<span class='badge badge-high'>🔴 High Risk</span>",
}

# ======================= RECOMMENDATIONS =======================
RECS = {
    "low risk": [
        "Continue routine antenatal care & BP checks.",
        "Educate on warning signs: severe headache, visual changes, epigastric pain, swelling, ↓ fetal movement.",
        "Healthy lifestyle: rest, hydration, balanced diet; monitor BP if available.",
        "Plan follow-up as per clinic protocol."
    ],
    "mid risk": [
        "Arrange **early follow-up** (e.g., 48–72h) to repeat BP and check proteinuria.",
        "Consider labs per local protocol (urinalysis/protein, CBC).",
        "Reinforce warning signs and a clear action plan.",
        "Review risk factors (HTN history, prior preeclampsia, BMI, Doppler indices) and optimize management."
    ],
    "high risk": [
        "**Prompt clinical evaluation advised.**",
        "Repeat/confirm BP; assess symptoms (headache, vision, RUQ pain).",
        "Check proteinuria & relevant labs per protocol.",
        "Consider fetal assessment and appropriate level of care/observation.",
        "Ensure the patient knows when to seek emergency care."
    ]
}
DISCLAIMER = ("This tool supports decisions; it does **not** replace clinical judgment. Follow local protocols and consult a clinician.")

# ======================= HELPERS & GUARDRAILS =======================
BIN_FEATS = ["History of hypertension (y/n)", "History of preeclampsia", "Protien Uria", "diabetes"]
INT_FEATS = ["Number of babies", "HeartRate"]
BP_CODE   = "Blood Pressure"  # optional coded feature (0/1/2) if present in training
DEFAULTS = {
    "gravida":1,"parity":0,"gestational age (weeks)":28,"Age (yrs)":27,"BMI  [kg/m²]":24.0,"diabetes":0,
    "History of hypertension (y/n)":0,"Systolic BP":120.0,"Diastolic BP":80.0,"HB":12.0,"fetal weight(kgs)":1.3,
    "Protien Uria":0,"Uterine Artery Doppler Resistance Index (RI)":0.6,"amniotic fluid levels(cm)":14.0,
    "History of preeclampsia":0,"Uterine Artery Doppler Pulsatility Index (PI)":1.1,"Number of babies":1,
    "BodyTemp":98.6,"HeartRate":80,"Blood Pressure":1,"BS":6.0,"Systolic_BP":120.0,"Diastolic_BP":80.0,
}
BOUNDS = {
    "gestational age (weeks)":(10,45),"Age (yrs)":(14,55),"BMI  [kg/m²]":(12.0,55.0),"Systolic BP":(80.0,200.0),
    "Diastolic BP":(40.0,120.0),"HB":(6.0,20.0),"fetal weight(kgs)":(0.3,5.0),
    "Uterine Artery Doppler Resistance Index (RI)":(0.2,1.2),"Uterine Artery Doppler Pulsatility Index (PI)":(0.5,2.5),
    "amniotic fluid levels(cm)":(2.0,30.0),"Number of babies":(1,3),"BodyTemp":(95.0,103.0),"HeartRate":(50,150),
    "BS":(3.0,20.0),"Systolic_BP":(80.0,200.0),"Diastolic_BP":(40.0,120.0),
}
def coerce_clip(name, val):
    if name in BOUNDS:
        lo, hi = BOUNDS[name]
        try: x = float(val)
        except: x = float(DEFAULTS.get(name, lo))
        x = float(np.clip(x, lo, hi))
        if name in INT_FEATS: x = int(x)
        return x
    if name in INT_FEATS:
        try: return int(val)
        except: return int(DEFAULTS.get(name, 0))
    try: return float(val)
    except: return float(DEFAULTS.get(name, 0.0))

def bp_to_code(sys_bp, dia_bp):
    if sys_bp < 100 or dia_bp < 60: return 0
    if sys_bp >= 140 or dia_bp >= 90: return 2
    return 1

def benign_profile(x):
    sys_bp = float(x.get("Systolic BP", x.get("Systolic_BP", 120)))
    dia_bp = float(x.get("Diastolic BP", x.get("Diastolic_BP", 80)))
    prot   = int(x.get("Protien Uria", 0))
    htn_hx = int(x.get("History of hypertension (y/n)", 0))
    pree_hx= int(x.get("History of preeclampsia", 0))
    bmi    = float(x.get("BMI  [kg/m²]", 24))
    pi     = float(x.get("Uterine Artery Doppler Pulsatility Index (PI)", 1.1))
    ri     = float(x.get("Uterine Artery Doppler Resistance Index (RI)", 0.6))
    ga     = float(x.get("gestational age (weeks)", 28))
    bs     = float(x.get("BS", 6.0))
    af     = float(x.get("amniotic fluid levels(cm)", 14.0))
    return all([
        100 <= sys_bp <= 139, 60 <= dia_bp <= 89,
        prot == 0, htn_hx == 0, pree_hx == 0,
        bmi < 30, pi <= 1.3, ri <= 0.7,
        12 <= ga <= 42, 5 <= af <= 24, bs <= 7.8
    ])

def severe_flags(x):
    sys_bp = float(x.get("Systolic BP", x.get("Systolic_BP", 120)))
    dia_bp = float(x.get("Diastolic BP", x.get("Diastolic_BP", 80)))
    prot   = int(x.get("Protien Uria", 0))
    pi     = float(x.get("Uterine Artery Doppler Pulsatility Index (PI)", 1.1))
    ri     = float(x.get("Uterine Artery Doppler Resistance Index (RI)", 0.6))
    return {
        "very_high_bp": (sys_bp >= 160 or dia_bp >= 110),
        "proteinuria": (prot == 1),
        "doppler_severe": (pi >= 1.6 or ri >= 0.8)
    }

def clinical_guardrails(raw_idx, proba, xrow, hi_cut=0.65, low_cut=0.55):
    raw_lbl = label_encoder.inverse_transform([raw_idx])[0]
    pmap = dict(zip(CLASSES, proba))
    if benign_profile(xrow) and pmap.get("high risk", 0.0) < hi_cut:
        return "low risk", "Benign profile → set to Low Risk."
    flags = severe_flags(xrow)
    n_severe = sum(flags.values())
    if raw_lbl == "low risk" and (flags["very_high_bp"] or flags["proteinuria"] or flags["doppler_severe"]):
        if pmap.get("low risk", 0.0) < low_cut:
            if n_severe >= 2 and pmap.get("mid risk", 0.0) < 0.60:
                return "high risk", "Multiple severe indicators → escalated to High."
            return "mid risk", "Severe indicator → escalated to Moderate."
    return raw_lbl, None

def prob_table(proba):
    return pd.DataFrame({"Class": CLASSES, "Probability": proba}).sort_values("Probability", ascending=False)

def show_recs(label):
    color = "panel-low" if label=="low risk" else "panel-mid" if label=="mid risk" else "panel-high"
    st.markdown(f"<div class='panel {color}'>", unsafe_allow_html=True)
    st.markdown("**Recommendations / Precautions**")
    for item in RECS.get(label, []):
        st.markdown(f"- {item}")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= UI =======================
tab1, tab2 = st.tabs(["🧍 Single Prediction", "📄 Batch CSV"])

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Patient Inputs")

    ui = {}
    c1, c2 = st.columns(2)
    for i, feat in enumerate(feature_names):
        col = c1 if i % 2 == 0 else c2
        default_val = DEFAULTS.get(feat, 0)

        if feat in BIN_FEATS:
            with col:
                ui[feat] = st.selectbox(feat, ["0","1"], index=int(str(default_val) in ["1","y","Y","yes","True","true"]))
        elif feat in INT_FEATS:
            lo, hi = BOUNDS.get(feat, (0, 999))
            with col:
                ui[feat] = st.number_input(feat, min_value=int(lo), max_value=int(hi), value=int(coerce_clip(feat, default_val)), step=1)
        elif feat == BP_CODE:
            with col:
                sel = st.selectbox(feat, ["low","normal","high"], index=1)
                ui[feat] = {"low":0,"normal":1,"high":2}[sel]
        else:
            if feat in BOUNDS:
                lo, hi = BOUNDS[feat]
                with col:
                    base = coerce_clip(feat, default_val)
                    if float(base).is_integer() and feat not in ["BMI  [kg/m²]", "BS", "HB"]:
                        ui[feat] = st.number_input(feat, min_value=int(lo), max_value=int(hi), value=int(base), step=1)
                    else:
                        ui[feat] = st.number_input(feat, min_value=float(lo), max_value=float(hi), value=float(base))
            else:
                with col:
                    try: ui[feat] = st.number_input(feat, value=float(default_val))
                    except: ui[feat] = st.number_input(feat, value=0.0)

    use_guard = st.checkbox("Use clinical guardrails (recommended)", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🤖 Predict"):
        # Build ordered row + coerce/clip
        row_vals = []
        for f in feature_names:
            v = ui.get(f, DEFAULTS.get(f, 0))
            if f in BIN_FEATS:
                v = 1 if str(v) in ["1","y","Y","yes","True","true"] else 0
                row_vals.append(int(v)); continue
            if f == BP_CODE:
                row_vals.append(int(v)); continue
            row_vals.append(coerce_clip(f, v))
        X = pd.DataFrame([row_vals], columns=feature_names)

        # Derive coded BP if needed
        if BP_CODE in X.columns:
            sys_bp = float(X.get("Systolic BP", X.get("Systolic_BP", pd.Series([120]))).iloc[0])
            dia_bp = float(X.get("Diastolic BP", X.get("Diastolic_BP", pd.Series([80]))).iloc[0])
            X.loc[0, BP_CODE] = bp_to_code(sys_bp, dia_bp)

        # Predict
        y_idx  = model.predict(X)[0]
        proba  = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
        label  = label_encoder.inverse_transform([y_idx])[0]
        note   = None
        if use_guard and proba is not None:
            label, note = clinical_guardrails(y_idx, proba, X.iloc[0])

        st.markdown(f"**Prediction:** {BADGE.get(label, label)}", unsafe_allow_html=True)
        if note: st.caption(f"ℹ️ {note}")

        if proba is not None:
            st.write("**Class probabilities**")
            dfp = prob_table(proba)
            for _, row in dfp.iterrows():
                st.markdown(f"<div class='prob-row'><span>{row['Class']}</span><span>{row['Probability']:.2%}</span></div>", unsafe_allow_html=True)

        show_recs(label)
        st.info(DISCLAIMER)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Batch Prediction (CSV)")
    st.caption("Columns aligned to training features. Binary fields auto-encoded. If present, coded ‘Blood Pressure’ is derived from systolic/diastolic.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            try:
                df = pd.read_csv(up)
            except UnicodeDecodeError:
                up.seek(0); df = pd.read_csv(up, encoding="latin1")

            for col in BIN_FEATS:
                if col in df.columns:
                    df[col] = df[col].replace({"n":0,"y":1,"0":0,"1":1}).infer_objects(copy=False)

            if BP_CODE in feature_names:
                if {"Systolic BP","Diastolic BP"}.issubset(df.columns) or {"Systolic_BP","Diastolic_BP"}.issubset(df.columns):
                    s_col = "Systolic BP" if "Systolic BP" in df.columns else "Systolic_BP"
                    d_col = "Diastolic BP" if "Diastolic BP" in df.columns else "Diastolic_BP"
                    df[BP_CODE] = [bp_to_code(coerce_clip("Systolic BP", s), coerce_clip("Diastolic BP", d)) for s, d in zip(df[s_col], df[d_col])]
                else:
                    df[BP_CODE] = 1  # default: normal

            missing = set(feature_names) - set(df.columns)
            extra   = set(df.columns) - set(feature_names)
            for c in missing: df[c] = 0
            if extra: df = df.drop(columns=list(extra))
            df = df[feature_names]

            yhat = model.predict(df)
            proba = model.predict_proba(df) if hasattr(model, "predict_proba") else None
            raw_labels = label_encoder.inverse_transform(yhat)

            final_labels = []
            if proba is not None:
                for i in range(len(df)):
                    lbl, _ = clinical_guardrails(yhat[i], proba[i], df.iloc[i])
                    final_labels.append(lbl)
            else:
                final_labels = raw_labels

            out = df.copy()
            out["prediction_raw"] = raw_labels
            out["prediction_final"] = final_labels

            st.success("✅ Batch predictions completed.")
            st.dataframe(out.tail(25), use_container_width=True)
            st.download_button("⬇️ Download predictions (CSV)",
                               data=out.to_csv(index=False).encode("utf-8"),
                               file_name="batch_predictions.csv",
                               mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
