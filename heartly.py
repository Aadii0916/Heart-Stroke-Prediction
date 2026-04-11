
import streamlit as st
import pandas as pd
import joblib
 
# ── Load model artifacts ──
model            = joblib.load("KNN_heart.pkl")
scaler           = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")
 
# ── Page config ──
st.set_page_config(
    page_title = "Heart Disease Risk AI",
    page_icon  = "🫀",
    layout     = "wide"
)
 
# ════════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
 
/* ── BACKGROUND: dark overlay on a cardiac MRI / anatomy image ── */
.stApp {
    background-color: #060c18;
    background-image:
        linear-gradient(rgba(4,8,18,0.80), rgba(4,8,18,0.87)),
        url("https://images.unsplash.com/photo-1628348068343-c6a848d2b6dd?w=1600&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #e2e8f0 !important;
}
 
/* ── Animated scanner line across the top ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent 0%, #3b82f6 50%, transparent 100%);
    animation: scanLine 3s linear infinite;
    z-index: 9999;
}
@keyframes scanLine {
    from { transform: translateX(-100%); }
    to   { transform: translateX(100%);  }
}
 
/* ── Dot-grid overlay for a clinical monitor feel ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(rgba(59,130,246,0.055) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
    z-index: 0;
}
 
/* ── Main glassmorphism card ── */
.main-container {
    position: relative;
    z-index: 1;
    max-width: 1020px;
    margin: 26px auto;
    padding: 30px 36px;
    background: rgba(5,12,28,0.80);
    border: 0.5px solid rgba(59,130,246,0.25);
    border-radius: 20px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}
 
/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(3,7,18,0.94) !important;
    border-right: 1px solid rgba(59,130,246,0.14);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
 
/* ── Labels ── */
label,
.stSelectbox label,
.stNumberInput label,
.stSlider label {
    color: #64748b !important;
    font-size: 12px !important;
}
 
/* ── Select boxes ── */
.stSelectbox > div > div {
    background: rgba(10,20,46,0.90) !important;
    border: 0.5px solid rgba(59,130,246,0.26) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    transition: border-color .2s, box-shadow .2s;
}
.stSelectbox > div > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.14) !important;
}
.stSelectbox svg { fill: #3b82f6 !important; }
 
/* ── Number inputs ── */
.stNumberInput > div > div > input {
    background: rgba(10,20,46,0.90) !important;
    border: 0.5px solid rgba(59,130,246,0.26) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    transition: border-color .2s, box-shadow .2s;
}
.stNumberInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.14) !important;
    outline: none !important;
}
 
/* ── Sliders ── */
.stSlider > div > div > div { background: #3b82f6 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #3b82f6 !important;
    border-color: #1d4ed8 !important;
    box-shadow: 0 0 10px rgba(59,130,246,0.5) !important;
}
.stSlider [data-baseweb="slider"] div[role="progressbar"] {
    background: linear-gradient(90deg, #1e3a8a, #3b82f6) !important;
}
 
/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #3b82f6 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: .8px !important;
    width: 100% !important;
    box-shadow: 0 4px 24px rgba(59,130,246,0.38) !important;
    transition: opacity .2s, transform .1s, box-shadow .2s !important;
}
.stButton > button:hover {
    opacity: .88 !important;
    box-shadow: 0 6px 32px rgba(59,130,246,0.55) !important;
}
.stButton > button:active { transform: scale(.985) !important; }
 
/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1e3a8a, #3b82f6, #60a5fa) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 4px !important;
}
 
/* ── Dividers ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #1e3a8a, #3b82f6, #1e3a8a, transparent) !important;
    margin: 16px 0 !important;
}
 
/* ── Section label ── */
.sec-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3b82f6;
    font-weight: 700;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 0.5px solid rgba(59,130,246,0.18);
}
 
/* ── Live meter box ── */
.meter-box {
    background: rgba(8,16,36,0.82);
    border: 0.5px solid rgba(59,130,246,0.25);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 18px;
}
 
/* ── Flag pills ── */
.flag-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}
.flag-danger {
    background: rgba(239,68,68,0.12);
    border: 0.5px solid rgba(239,68,68,0.35);
    color: #fca5a5;
}
.flag-ok {
    background: rgba(34,197,94,0.09);
    border: 0.5px solid rgba(34,197,94,0.28);
    color: #86efac;
}
 
/* ── Result cards ── */
.result-high {
    padding: 26px 28px;
    border-radius: 16px;
    background: rgba(127,29,29,0.20);
    border: 1px solid rgba(239,68,68,0.44);
    margin-top: 22px;
    box-shadow: 0 0 40px rgba(239,68,68,0.12), inset 0 0 40px rgba(127,29,29,0.07);
    animation: slideUp .45s cubic-bezier(.4,0,.2,1);
}
.result-low {
    padding: 26px 28px;
    border-radius: 16px;
    background: rgba(6,78,59,0.17);
    border: 1px solid rgba(34,197,94,0.36);
    margin-top: 22px;
    box-shadow: 0 0 40px rgba(34,197,94,0.10), inset 0 0 40px rgba(6,78,59,0.07);
    animation: slideUp .45s cubic-bezier(.4,0,.2,1);
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
 
/* ── Metric chips ── */
.chip-row {
    display: flex;
    gap: 9px;
    flex-wrap: wrap;
    margin: 14px 0;
    justify-content: center;
}
.chip {
    background: rgba(255,255,255,0.055);
    border: 0.5px solid rgba(255,255,255,0.10);
    border-radius: 9px;
    padding: 8px 14px;
    font-size: 11px;
    color: #94a3b8;
    text-align: center;
}
.chip span { display:block; font-size:15px; font-weight:700; color:#e2e8f0; margin-top:2px; }
.chip small { display:block; font-size:10px; margin-top:1px; }
 
/* ── Two-column tips grid ── */
.tips-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 14px;
}
.tips-box {
    padding: 14px 18px;
    background: rgba(30,58,138,0.14);
    border: 0.5px solid rgba(59,130,246,0.20);
    border-radius: 10px;
    font-size: 12px;
    color: #7cb3f0;
    line-height: 1.85;
}
.tips-box ul { margin: 8px 0 0 14px; }
 
/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #060c18; }
::-webkit-scrollbar-thumb { background: #1e3a8a; border-radius: 4px; }
 
/* ── Heartbeat keyframe ── */
@keyframes hbBeat {
    0%,100% { transform: scale(1); }
    15%     { transform: scale(1.28); }
    30%     { transform: scale(1); }
}
/* ── Blink dot ── */
@keyframes blinkDot {
    0%,100% { opacity: 1; }
    50%     { opacity: .25; }
}
 
</style>
""", unsafe_allow_html=True)
 
 
# ════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════
def heuristic_score(age, max_hr, oldpeak, angina, slope, cp, fbs, chol, bp):
    s = 10
    s += max(0, (age - 18) * 0.5)
    s += max(0, (220 - max_hr) * 0.2)
    s += oldpeak * 5
    s += max(0, (chol - 200) * 0.05)
    s += max(0, (bp - 120) * 0.18)
    if angina == "Y":         s += 15
    if slope == "Down":       s += 12
    elif slope == "Flat":     s += 6
    if cp == "ASY":           s += 12
    elif cp == "TA":          s += 6
    if fbs == 1:              s += 8
    return min(95, max(5, round(s)))
 
 
def flag_pill(label, danger, note=""):
    cls   = "flag-danger" if danger else "flag-ok"
    color = "#ef4444"     if danger else "#22c55e"
    icon  = "⚠"           if danger else "✓"
    suffix = f" — {note}" if note else ""
    return (
        f'<span class="flag-pill {cls}">'
        f'<span style="color:{color};font-size:12px;">{icon}</span>'
        f'{label}{suffix}</span>'
    )
 
 
# ════════════════════════════════════════════
#  OPEN MAIN CONTAINER
# ════════════════════════════════════════════
st.markdown('<div class="main-container">', unsafe_allow_html=True)
 
# ── Animated ECG stripe ──
st.markdown("""
<div style="width:100%;height:36px;overflow:hidden;opacity:0.42;margin-bottom:6px;">
  <svg viewBox="0 0 1000 36" preserveAspectRatio="none" width="100%" height="36"
       xmlns="http://www.w3.org/2000/svg">
    <defs>
      <style>
        .ecg-s {
          stroke:#3b82f6; stroke-width:1.8; fill:none;
          stroke-dasharray:1000; stroke-dashoffset:1000;
          animation:ecgDraw 2.2s linear infinite;
        }
        @keyframes ecgDraw { to { stroke-dashoffset:-1000; } }
      </style>
    </defs>
    <polyline class="ecg-s"
      points="0,18 80,18 100,18 115,3 130,33 145,3 165,18 260,18
              340,18 355,3 370,33 385,3 405,18 500,18
              580,18 595,3 610,33 625,3 645,18 740,18
              820,18 835,3 850,33 865,3 885,18 1000,18"/>
  </svg>
</div>
""", unsafe_allow_html=True)
 
# ── Header ──
c_icon, c_title = st.columns([1, 14])
with c_icon:
    st.markdown("""
    <div style="margin-top:10px;">
      <svg width="46" height="42" viewBox="0 0 46 42" fill="none">
        <path d="M23 39C23 39 2 25 2 12.5C2 6.5 7.2 2 13.5 2C18.3 2 22.1 5 23 9
                 C23.9 5 27.7 2 32.5 2C38.8 2 44 6.5 44 12.5C44 25 23 39 23 39Z"
              fill="#2563eb" stroke="#1d4ed8" stroke-width="1.4"/>
        <path d="M9,21 L14,21 L17,12 L20,30 L23,16 L26,26 L29,21 L37,21"
              stroke="#93c5fd" stroke-width="1.9" fill="none"
              stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
      </svg>
    </div>
    """, unsafe_allow_html=True)
 
with c_title:
    st.markdown("""
    <div style="padding-top:4px;">
      <span style="font-size:23px;font-weight:800;color:#ffffff;">
        Heart Disease Risk Prediction AI
      </span>
      <span style="display:inline-flex;align-items:center;gap:5px;
                   background:rgba(59,130,246,0.15);border:0.5px solid rgba(59,130,246,0.35);
                   border-radius:20px;padding:3px 11px;font-size:11px;color:#93c5fd;
                   margin-left:12px;vertical-align:middle;">
        <span style="width:7px;height:7px;border-radius:50%;background:#3b82f6;
                     display:inline-block;animation:blinkDot 1.2s ease-in-out infinite;"></span>
        KNN Model Active
      </span>
    </div>
    <p style="color:#475569;font-size:12px;margin-top:5px;margin-bottom:0;">
      Adjust the inputs — the <b style="color:#93c5fd;">live risk score</b> and
      <b style="color:#93c5fd;">health flags</b> update instantly.
      Click <b style="color:#93c5fd;">Run Assessment</b> to invoke the trained model.
    </p>
    """, unsafe_allow_html=True)
 
st.write("---")
 
# ════════════════════════════════════════════
#  INPUT WIDGETS
# ════════════════════════════════════════════
col1, col2 = st.columns(2, gap="large")
 
with col1:
    st.markdown('<div class="sec-label">👤 Patient Demographics</div>', unsafe_allow_html=True)
    age         = st.slider("Age", 18, 100, 40)
    sex         = st.selectbox("Sex", ["M", "F"])
    chest_pain  = st.selectbox(
        "Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
        help="ATA = Atypical Angina · NAP = Non-Anginal Pain · TA = Typical Angina · ASY = Asymptomatic"
    )
    resting_bp  = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120,
                                  help="Normal: 90–120 mm Hg | Hypertensive: ≥ 140")
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200,
                                  help="Desirable: < 200 | Borderline: 200–239 | High: ≥ 240")
 
with col2:
    st.markdown('<div class="sec-label">🏥 Clinical Indicators</div>', unsafe_allow_html=True)
    fasting_bs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                                   format_func=lambda x: "Yes" if x == 1 else "No")
    resting_ecg     = st.selectbox(
        "Resting ECG Result", ["Normal", "ST", "LVH"],
        help="ST = ST-T wave abnormality · LVH = Left Ventricular Hypertrophy"
    )
    max_hr          = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150)
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina", ["N", "Y"],
        format_func=lambda x: "Yes" if x == "Y" else "No"
    )
    oldpeak         = st.slider("Oldpeak — ST Depression", 0.0, 6.0, 1.0, step=0.1,
                                help="ST depression induced by exercise relative to rest")
    st_slope        = st.selectbox(
        "ST Slope", ["Up", "Flat", "Down"],
        help="Slope of the peak exercise ST segment"
    )
 
 
# ════════════════════════════════════════════
#  LIVE RISK METER
# ════════════════════════════════════════════
live = heuristic_score(
    age, max_hr, oldpeak, exercise_angina,
    st_slope, chest_pain, fasting_bs, cholesterol, resting_bp
)
 
if live > 65:
    bar_grad   = "linear-gradient(90deg,#1e3a8a,#ef4444)"
    num_color  = "#f87171"
    tier_label = "High Risk"
    tier_bg    = "rgba(239,68,68,0.14)"
    tier_color = "#f87171"
    tier_bdr   = "rgba(239,68,68,0.35)"
elif live > 40:
    bar_grad   = "linear-gradient(90deg,#1e3a8a,#fb923c)"
    num_color  = "#fb923c"
    tier_label = "Moderate"
    tier_bg    = "rgba(251,146,60,0.14)"
    tier_color = "#fb923c"
    tier_bdr   = "rgba(251,146,60,0.35)"
else:
    bar_grad   = "linear-gradient(90deg,#1e3a8a,#4ade80)"
    num_color  = "#4ade80"
    tier_label = "Low Risk"
    tier_bg    = "rgba(34,197,94,0.12)"
    tier_color = "#4ade80"
    tier_bdr   = "rgba(34,197,94,0.30)"
 
st.markdown(f"""
<div class="meter-box">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <span style="font-size:10px;letter-spacing:2px;text-transform:uppercase;
                 color:#3b82f6;font-weight:700;">Live Risk Score</span>
    <span style="background:{tier_bg};border:0.5px solid {tier_bdr};color:{tier_color};
                 font-size:12px;font-weight:700;padding:3px 13px;border-radius:20px;">
      {tier_label}
    </span>
  </div>
  <div style="height:12px;background:rgba(255,255,255,0.07);border-radius:6px;
              overflow:hidden;margin-bottom:8px;">
    <div style="height:100%;width:{live}%;background:{bar_grad};border-radius:6px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:10px;
              color:#334155;margin-bottom:10px;">
    <span>Safe</span><span>Moderate</span><span>Critical</span>
  </div>
  <div style="display:flex;align-items:baseline;gap:8px;">
    <span style="font-size:13px;color:#ef4444;display:inline-block;
                 animation:hbBeat 1.2s ease-in-out infinite;">♥</span>
    <span style="font-size:34px;font-weight:800;color:{num_color};line-height:1;">{live}</span>
    <span style="font-size:13px;color:#475569;">/ 100 risk score</span>
  </div>
</div>
""", unsafe_allow_html=True)
 
 
# ════════════════════════════════════════════
#  HEALTH FLAG INDICATORS
# ════════════════════════════════════════════
flags_html = (
    flag_pill(f"Age {age} yrs",           age >= 60,                "Elevated" if age >= 60 else "")
  + flag_pill(f"Max HR {max_hr} bpm",     max_hr < 100,             "Low max HR" if max_hr < 100 else "")
  + flag_pill(f"Oldpeak {oldpeak}",        oldpeak >= 2.0,           "High ST depression" if oldpeak >= 2.0 else "")
  + flag_pill(f"Chol {cholesterol}",       cholesterol >= 240,       "High" if cholesterol >= 240 else "")
  + flag_pill(f"BP {resting_bp} mmHg",    resting_bp >= 140,        "Hypertensive" if resting_bp >= 140 else "")
  + flag_pill(f"Angina: {'Yes' if exercise_angina=='Y' else 'No'}",
              exercise_angina == "Y",      "Present" if exercise_angina == "Y" else "")
  + flag_pill(f"Slope: {st_slope}",        st_slope in ("Flat","Down"),
              "Abnormal" if st_slope in ("Flat","Down") else "")
  + flag_pill(f"CP: {chest_pain}",         chest_pain == "ASY",
              "High-risk type" if chest_pain == "ASY" else "")
  + flag_pill(f"FBS: {'High' if fasting_bs==1 else 'Normal'}", fasting_bs == 1,
              "> 120 mg/dL" if fasting_bs == 1 else "")
)
 
st.markdown(
    f'<div style="margin-bottom:18px;line-height:2.2;">{flags_html}</div>',
    unsafe_allow_html=True
)
 
 
# ════════════════════════════════════════════
#  PREDICT BUTTON
# ════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🫀  RUN CARDIAC RISK ASSESSMENT", use_container_width=True)
 
if predict_clicked:
 
    # ── Build input DataFrame ──
    raw_input = {
        'Age':                           age,
        'RestingBP':                     resting_bp,
        'Cholesterol':                   cholesterol,
        'FastingBS':                     fasting_bs,
        'MaxHR':                         max_hr,
        'Oldpeak':                       oldpeak,
        'Sex_'            + sex:                 1,
        'ChestPainType_'  + chest_pain:          1,
        'RestingECG_'     + resting_ecg:         1,
        'ExerciseAngina_' + exercise_angina:      1,
        'ST_Slope_'       + st_slope:            1,
    }
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df     = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction   = model.predict(scaled_input)[0]
    probability  = model.predict_proba(scaled_input)[0][1]
 
    if probability >= 0.75:   risk_tier = "Very High"
    elif probability >= 0.50: risk_tier = "High"
    elif probability >= 0.25: risk_tier = "Moderate"
    else:                     risk_tier = "Low"
 
    # ── Active danger flags list ──
    danger_list = []
    if age >= 60:                           danger_list.append(f"Age {age} yrs — elevated baseline risk")
    if max_hr < 100:                        danger_list.append(f"Low max HR ({max_hr} bpm) — reduced cardiac reserve")
    if oldpeak >= 2.0:                      danger_list.append(f"High oldpeak ({oldpeak}) — significant ST depression")
    if cholesterol >= 240:                  danger_list.append(f"High cholesterol ({cholesterol} mg/dL)")
    if resting_bp >= 140:                   danger_list.append(f"Hypertensive BP ({resting_bp} mmHg)")
    if exercise_angina == "Y":              danger_list.append("Exercise-induced angina present")
    if st_slope in ("Flat", "Down"):        danger_list.append(f"Abnormal ST slope ({st_slope})")
    if chest_pain == "ASY":                 danger_list.append("Asymptomatic chest pain (high-risk type)")
    if fasting_bs == 1:                     danger_list.append("Elevated fasting blood sugar (> 120 mg/dL)")
 
    # ── Metric chips ──
    def chip(label, value, note, bad):
        c = "#f87171" if bad else "#4ade80"
        return (
            f'<div class="chip">{label}'
            f'<span>{value}</span>'
            f'<small style="color:{c};">{note}</small></div>'
        )
 
    chips = (
        chip("Age",          f"{age} yrs",         "≥60 risk" if age >= 60 else "OK",                   age >= 60)
      + chip("Max HR",       f"{max_hr} bpm",       "Low" if max_hr < 100 else "OK",                    max_hr < 100)
      + chip("Cholesterol",  f"{cholesterol}",      "High" if cholesterol >= 240 else "OK",              cholesterol >= 240)
      + chip("Oldpeak",      f"{oldpeak}",           "High" if oldpeak >= 2 else "OK",                   oldpeak >= 2)
      + chip("BP",           f"{resting_bp} mmHg",  "Hypertensive" if resting_bp >= 140 else "OK",       resting_bp >= 140)
      + chip("Angina",       "Yes" if exercise_angina=="Y" else "No",
             "Present" if exercise_angina=="Y" else "None",                                              exercise_angina=="Y")
    )
 
    st.write("---")
 
    # ══════════════  HIGH RISK  ══════════════
    if prediction == 1:
 
        flags_ul = "".join(f"<li>{f}</li>" for f in danger_list) if danger_list \
                   else "<li>Multiple combined risk factors</li>"
 
        st.markdown(f"""
        <div class="result-high">
          <div style="text-align:center;margin-bottom:16px;">
            <div style="font-size:42px;margin-bottom:8px;">⚠️</div>
            <h2 style="color:#f87171;font-size:22px;font-weight:800;margin-bottom:5px;">
              Elevated Heart Disease Risk Detected
            </h2>
            <p style="color:#fca5a5;font-size:15px;margin-bottom:3px;">
              Model probability: <b>{probability:.2%}</b>
              &nbsp;|&nbsp; Risk tier: <b>{risk_tier}</b>
            </p>
            <p style="color:#64748b;font-size:12px;">
              The KNN classifier identifies a high-risk pattern across this patient's combined clinical profile.
            </p>
          </div>
          <div class="chip-row">{chips}</div>
          <div class="tips-grid">
            <div class="tips-box">
              <b style="color:#93c5fd;">Active Risk Factors</b>
              <ul>{flags_ul}</ul>
            </div>
            <div class="tips-box">
              <b style="color:#93c5fd;">Recommended Next Steps</b>
              <ul>
                <li>Consult a cardiologist promptly</li>
                <li>Schedule a stress ECG / echocardiogram</li>
                <li>Review full lipid panel</li>
                <li>Evaluate blood pressure management</li>
                <li>Heart-healthy diet + 150 min/week exercise</li>
                <li>Avoid smoking and limit alcohol</li>
              </ul>
            </div>
          </div>
          <p style="font-size:11px;color:#334155;text-align:center;margin-top:14px;">
            ⚕ This is the heart prediction.
          </p>
        </div>
        """, unsafe_allow_html=True)
 
    # ══════════════  LOW RISK  ══════════════
    else:
        st.markdown(f"""
        <div class="result-low">
          <div style="text-align:center;margin-bottom:16px;">
            <div style="font-size:42px;margin-bottom:8px;">✅</div>
            <h2 style="color:#4ade80;font-size:22px;font-weight:800;margin-bottom:5px;">
              Low Heart Disease Risk
            </h2>
            <p style="color:#86efac;font-size:15px;margin-bottom:3px;">
              Model probability: <b>{probability:.2%}</b>
              &nbsp;|&nbsp; Risk tier: <b>{risk_tier}</b>
            </p>
            <p style="color:#64748b;font-size:12px;">
              The KNN classifier finds no dominant high-risk pattern in this patient's profile.
            </p>
          </div>
          <div class="chip-row">{chips}</div>
          <div class="tips-grid">
            <div class="tips-box">
              <b style="color:#93c5fd;">Healthy Indicators</b>
              <ul>
                <li>Resting heart metrics within normal range</li>
                <li>ST segment within acceptable limits</li>
                <li>No exercise-induced angina symptoms</li>
                <li>Age, HR, and slope within safe profile</li>
              </ul>
            </div>
            <div class="tips-box">
              <b style="color:#93c5fd;">Staying Heart-Healthy</b>
              <ul>
                <li>Annual cardiac checkup recommended</li>
                <li>150 min/week moderate aerobic activity</li>
                <li>Diet low in saturated fat and sodium</li>
                <li>Monitor BP and cholesterol yearly</li>
                <li>Avoid smoking and limit alcohol</li>
              </ul>
            </div>
          </div>
          <p style="font-size:11px;color:#334155;text-align:center;margin-top:14px;">
            ✔ Low risk today doesn't mean zero risk — regular monitoring is essential.
          </p>
        </div>
        """, unsafe_allow_html=True)
 
    # ── Probability bar ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#475569;font-size:12px;margin-bottom:4px;">'
        'KNN model — heart disease probability score</p>',
        unsafe_allow_html=True
    )
    st.progress(float(probability))
    p_color = "#f87171" if probability > 0.5 else "#4ade80"
    st.markdown(
        f'<p style="color:{p_color};font-size:14px;font-weight:800;margin-top:4px;">'
        f'{probability:.2%} probability of heart disease</p>',
        unsafe_allow_html=True
    )
 
# ── Close container ──
st.markdown('</div>', unsafe_allow_html=True)
 