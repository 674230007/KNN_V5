import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ---------------------------------------------------------
# 1. ตั้งค่าหน้าเว็บ & Custom CSS เพื่อความสวยงาม
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease KNN Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff4b4b 0%, #900c3f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    .result-card-danger {
        background-color: #ffe6e6;
        border-left: 6px solid #ff4b4b;
        padding: 1.5rem;
        border-radius: 10px;
        color: #7a0000;
    }
    .result-card-safe {
        background-color: #e6fffa;
        border-left: 6px solid #00b894;
        padding: 1.5rem;
        border-radius: 10px;
        color: #004d3d;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. โหลดข้อมูล (Cache เพื่อความเร็วบน Streamlit Cloud)
# ---------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("Heart3.csv")

# ---------------------------------------------------------
# 3. Sidebar: การตั้งค่าโมเดล KNN
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=65)
    st.title("⚙️ ตั้งค่าโมเดล KNN")
    
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV (ไม่บังคับ)", type=["csv"])
    
    st.markdown("---")
    st.subheader("🔧 พารามิเตอร์ของโมเดล")
    k_value = st.slider("จำนวนเพื่อนบ้าน (K Neighbors)", min_value=1, max_value=25, value=7, step=2)
    metric_type = st.selectbox("วิธีวัดระยะห่าง (Distance Metric)", ["euclidean", "manhattan", "minkowski"])
    weights_type = st.selectbox("การถ่วงน้ำหนัก (Weights)", ["uniform", "distance"])
    scaler_type = st.selectbox("การปรับสเกลข้อมูล (Feature Scaling)", ["StandardScaler (แนะนำ)", "MinMaxScaler", "ไม่ปรับสเกล"])
    test_size = st.slider("สัดส่วนข้อมูลทดสอบ (Test Size %)", 10, 40, 20, step=5) / 100.0
    
    handle_zero = st.checkbox("แทนที่ค่า 0 ใน Cholesterol ด้วยค่ามัธยฐาน (Median)", value=False)

# โหลดและเตรียมข้อมูล
try:
    df_raw = load_data(uploaded_file)
except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์ `Heart3.csv` กรุณาวางไฟล์ไว้ในโฟลเดอร์เดียวกับ `app.py` หรืออัปโหลดผ่านแถบด้านซ้าย")
    st.stop()

df = df_raw.copy()
if handle_zero:
    median_chol = df[df["Cholesterol"] > 0]["Cholesterol"].median()
    df["Cholesterol"] = df["Cholesterol"].replace(0, median_chol)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# Scaling
if "StandardScaler" in scaler_type:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
elif "MinMaxScaler" in scaler_type:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    scaler = None
    X_train_scaled = X_train.values
    X_test_scaled = X_test.values

# เทรนโมเดล KNN
knn = KNeighborsClassifier(n_neighbors=k_value, metric=metric_type, weights=weights_type)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# คำนวณ Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ---------------------------------------------------------
# 4. Header หลักของแอป
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🫀 ระบบวิเคราะห์และจำแนกความเสี่ยงโรคหัวใจ (KNN Classifier)</h1>
    <p>ขับเคลื่อนด้วยอัลกอริทึม K-Nearest Neighbors บนชุดข้อมูล Heart3.csv (918 รายการ)</p>
</div>
""", unsafe_allow_html=True)

# แสดง KPI Summary ด้านบนสุด
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 ความแม่นยำ (Accuracy)", f"{acc*100:.2f}%")
col2.metric("🔍 ความเที่ยงตรง (Precision)", f"{prec*100:.2f}%")
col3.metric("📡 ความครอบคลุม (Recall)", f"{rec*100:.2f}%")
col4.metric("⚖️ F1-Score", f"{f1*100:.2f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. แบ่งหน้าจอเป็น 3 แท็บหลัก
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🩺 ทำนายความเสี่ยงรายบุคคล (Predict)",
    "📊 ประเมินผลโมเดล & หาค่า K (Model Performance)",
    "🗂️ สำรวจชุดข้อมูล (Data Explorer)"
])

# ==================== TAB 1: PREDICTION ====================
with tab1:
    st.subheader("กรอกข้อมูลสุขภาพของผู้ป่วยเพื่อประเมินความเสี่ยง")
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.number_input("อายุ (Age - ปี)", min_value=18, max_value=100, value=54)
            sex = st.selectbox("เพศ (Sex)", options=[1, 0], format_func=lambda x: "1 : ชาย (Male)" if x == 1 else "0 : หญิง (Female)")
            cp = st.selectbox("ประเภทอาการเจ็บหน้าอก (ChestPainType)", options=[1, 2, 3, 4],
                              format_func=lambda x: {
                                  1: "1 : ASY (ไม่มีอาการเจ็บหน้าอก)",
                                  2: "2 : ATA (เจ็บหน้าอกแบบไม่คงที่)",
                                  3: "3 : NAP (เจ็บหน้าอกที่ไม่ใช่โรคหัวใจ)",
                                  4: "4 : TA (เจ็บหน้าอกแบบทั่วไป)"
                              }[x])
            resting_bp = st.number_input("ความดันโลหิตขณะพัก (RestingBP - mm Hg)", min_value=60, max_value=250, value=130)

        with c2:
            cholesterol = st.number_input("ระดับคอเลสเตอรอล (Cholesterol - mg/dl)", min_value=0, max_value=650, value=220)
            fasting_bs = st.selectbox("น้ำตาลในเลือดหลังอดอาหาร > 120 mg/dl (FastingBS)", options=[0, 1],
                                      format_func=lambda x: "1 : ใช่ (> 120 mg/dl)" if x == 1 else "0 : ไม่ใช่ (ปกติ)")
            resting_ecg = st.selectbox("คลื่นไฟฟ้าหัวใจขณะพัก (RestingECG)", options=[1, 2, 3],
                                       format_func=lambda x: {1: "1 : LVH", 2: "2 : Normal (ปกติ)", 3: "3 : ST"}[x])
            max_hr = st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR - bpm)", min_value=50, max_value=220, value=140)

        with c3:
            ex_angina = st.selectbox("เจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina)", options=[0, 1],
                                     format_func=lambda x: "1 : มีอาการ (Yes)" if x == 1 else "0 : ไม่มีอาการ (No)")
            oldpeak = st.number_input("ค่า Oldpeak (ST Depression)", min_value=-3.0, max_value=7.0, value=0.5, step=0.1)
            st_slope = st.selectbox("ความชันคลื่น ST (ST_Slope)", options=[1, 2, 3],
                                    format_func=lambda x: {1: "1 : Down (ลาดลง)", 2: "2 : Flat (แบนราบ)", 3: "3 : Up (ลาดขึ้น)"}[x])

        submitted = st.form_submit_button("🔍 วิเคราะห์ผลการจำแนกข้อมูล", use_container_width=True, type="primary")

    if submitted:
        input_data = np.array([[age, sex, cp, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, ex_angina, oldpeak, st_slope]])
        if scaler is not None:
            input_data = scaler.transform(input_data)
        
        prediction = knn.predict(input_data)[0]
        proba = knn.predict_proba(input_data)[0]
        
        st.markdown("### ผลการวิเคราะห์จากเพื่อนบ้านใกล้เคียงที่สุด (K = {})".format(k_value))
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card-danger">
                    <h2>⚠️ พบความเสี่ยงภาวะโรคหัวใจ (HeartDisease = 1)</h2>
                    <p>โมเดลประเมินว่ามีโอกาสเป็นโรคหัวใจ <b>{proba[1]*100:.1f}%</b> (จากกลุ่มเพื่อนบ้านที่ใกล้เคียงที่สุด {k_value} ราย)<br>
                    <i>คำแนะนำ: ควรปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อตรวจวินิจฉัยอย่างละเอียด</i></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-safe">
                    <h2>✅ ความเสี่ยงต่ำ / ปกติ (HeartDisease = 0)</h2>
                    <p>โมเดลประเมินว่าอยู่ในเกณฑ์ปกติ <b>{proba[0]*100:.1f}%</b> (จากกลุ่มเพื่อนบ้านที่ใกล้เคียงที่สุด {k_value} ราย)<br>
                    <i>คำแนะนำ: ควรรักษาสุขภาพและตรวจเช็คร่างกายเป็นประจำทุกปี</i></p>
                </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1] * 100,
                title={'text': "ความน่าจะเป็นโรคหัวใจ (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4b4b" if prediction == 1 else "#00b894"},
                    'steps': [
                        {'range': [0, 40], 'color': "#e6fffa"},
                        {'range': [40, 70], 'color': "#fff3cd"},
                        {'range': [70, 100], 'color': "#ffe6e6"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

# ==================== TAB 2: MODEL EVALUATION ====================
with tab2:
    eval_col1, eval_col2 = st.columns(2)
    
    with eval_col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="ค่าที่โมเดลทำนาย (Predicted)", y="ค่าจริง (Actual)", color="จำนวนคน"),
            x=["ปกติ (0)", "โรคหัวใจ (1)"],
            y=["ปกติ (0)", "โรคหัวใจ (1)"],
            color_continuous_scale="Reds"
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with eval_col2:
        st.subheader("กราฟเปรียบเทียบความแม่นยำตามค่า K (K = 1 ถึง 25)")
        k_range = list(range(1, 26, 2))
        accuracies = []
        for k in k_range:
            temp_knn = KNeighborsClassifier(n_neighbors=k, metric=metric_type, weights=weights_type)
            temp_knn.fit(X_train_scaled, y_train)
            accuracies.append(accuracy_score(y_test, temp_knn.predict(X_test_scaled)) * 100)
            
        df_k = pd.DataFrame({"K": k_range, "Accuracy (%)": accuracies})
        fig_k = px.line(df_k, x="K", y="Accuracy (%)", markers=True, line_shape="spline")
        fig_k.add_vline(x=k_value, line_dash="dash", line_color="red", annotation_text=f"K ปัจจุบัน ({k_value})")
        fig_k.update_traces(line_color="#900c3f", marker_size=8)
        fig_k.update_layout(height=400)
        st.plotly_chart(fig_k, use_container_width=True)

# ==================== TAB 3: DATA EXPLORER ====================
with tab3:
    st.subheader("ภาพรวมชุดข้อมูล Heart3.csv")
    eda_col1, eda_col2 = st.columns(2)
    
    with eda_col1:
        fig_target = px.pie(
            df, names="HeartDisease", title="สัดส่วนผู้ป่วยโรคหัวใจ (1 = เป็น, 0 = ปกติ)",
            color="HeartDisease", color_discrete_map={0: "#00b894", 1: "#ff4b4b"},
            hole=0.4
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
    with eda_col2:
        selected_feat = st.selectbox("เลือกตัวแปรเพื่อดูการกระจายตัวเทียบกับโรคหัวใจ", ["Age", "MaxHR", "Cholesterol", "RestingBP", "Oldpeak"])
        fig_box = px.box(
            df, x="HeartDisease", y=selected_feat, color="HeartDisease",
            title=f"การกระจายตัวของ {selected_feat} แยกตามกลุ่มโรคหัวใจ",
            color_discrete_map={0: "#00b894", 1: "#ff4b4b"}
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.markdown("#### ตัวอย่างข้อมูล 20 แถวแรก")
    st.dataframe(df.head(20), use_container_width=True)