from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("โปรเจคการจำแนกข้อมูลโรคหัวใจ")

# สามารถนำรูปหัวใจมาใส่ในโฟลเดอร์ img แล้วเปิดใช้งานบรรทัดด้านล่างได้
# st.image("./img/heart.jpg")
col1, col2, col3 = st.columns(3)

with col1:
   st.header("Heart")
   st.image("./img/Heart1.jpg")

with col2:
   st.header("Heart") # อย่าลืมเปลี่ยนหัวข้อเป็นเรื่องโรคหัวใจด้วยนะครับ
   st.image("./img/Heart2.png")

with col3:
   st.header("profile")
   st.image("./img/profile.jpg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>สถิติข้อมูลโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

# คำนวณค่าเฉลี่ยของข้อมูลสำคัญเพื่อแสดงกราฟแท่ง
dt1 = dt['Age'].mean()
dt2 = dt['RestingBP'].mean()
dt3 = dt['Cholesterol'].mean()
dt4 = dt['MaxHR'].mean()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["Age (อายุ)", "RestingBP (ความดัน)", "Cholesterol (ไขมัน)", "MaxHR (ชีพจรสูงสุด)"], columns=["ค่าเฉลี่ย"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
    st.bar_chart(dx2)
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px;border-style:solid;border-color:black">
<center><h5>ทำนายความเสี่ยงโรคหัวใจ</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

# รับค่าอินพุตให้ครบ 11 คอลัมน์ตามไฟล์ Heart3.csv
col1, col2 = st.columns(2)

with col1:
   age = st.number_input("Age (อายุ)", min_value=1, max_value=120, value=40)
   sex = st.selectbox("Sex (เพศ: 1=ชาย, 0=หญิง)", [1, 0])
   chest_pain = st.selectbox("ChestPainType (ประเภทอาการเจ็บหน้าอก: 1-4)", [1, 2, 3, 4])
   resting_bp = st.number_input("RestingBP (ความดันโลหิตขณะพัก)", min_value=0, max_value=300, value=140)
   cholesterol = st.number_input("Cholesterol (ระดับคอเลสเตอรอล)", min_value=0, max_value=600, value=200)
   fasting_bs = st.selectbox("FastingBS (น้ำตาลในเลือด > 120: 1=ใช่, 0=ไม่ใช่)", [0, 1])

with col2:
    resting_ecg = st.selectbox("RestingECG (คลื่นไฟฟ้าหัวใจขณะพัก: 0-2)", [0, 1, 2])
    max_hr = st.number_input("MaxHR (อัตราการเต้นหัวใจสูงสุด)", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("ExerciseAngina (เจ็บหน้าอกขณะออกกำลังกาย: 1=ใช่, 0=ไม่ใช่)", [0, 1])
    oldpeak = st.number_input("Oldpeak (ค่าความผิดปกติของคลื่น ST)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
    st_slope = st.selectbox("ST_Slope (ความชันของคลื่น ST: 1-3)", [1, 2, 3])

if st.button("ทำนายผล"):
    X = dt.drop('HeartDisease', axis=1)
    y = dt['HeartDisease']

    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)

    # เรียงลำดับข้อมูลนำเข้า 11 ตัวให้ตรงกับคอลัมน์ใน Heart3.csv
    x_input = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

    out = Knn_model.predict(x_input)

    if out[0] == 1:
        st.error("ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ (HeartDisease = 1)")
        # st.image("./img/heart_sick.jpg")  # เปิดใช้หากมีรูปภาพประกอบ
    else:
        st.success("ผลการทำนาย: ปกติ ไม่มีความเสี่ยงโรคหัวใจ (HeartDisease = 0)")
        # st.image("./img/heart_normal.jpg") # เปิดใช้หากมีรูปภาพประกอบ
else:
    st.write("ยังไม่ทำนาย")