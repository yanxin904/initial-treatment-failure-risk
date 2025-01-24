#以下内容来源：微信公众号一篇-收藏夹里找
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load(r'D:\project\重症肺炎模型预处理、SMOTE、建模、验证等\XGB10.joblib')

# 如果有分类变量，需要定义特征选项
#Quinolone_options = {
#    1: 'use',
#    0: 'not use'
#}

# Define feature names
feature_names = [
    "LDH", "Mb","CK-MB","PCT", "T", "P","CRP","RDW-SD"
]

# Streamlit user interface
st.title("Failure of initial treatment for severe pneumonia")

#设置变量输入
LDH = st.number_input("LDH:", min_value=100.0, max_value=3000.0, value=300.0) #value是默认值的意思，选择中位数
Mb = st.number_input("Mb:", min_value=20.0, max_value=4000.0, value=180.0)
CK_MB = st.number_input("CK-MB:", min_value=1.0, max_value=260.0, value=18.0)
PCT = st.number_input("PCT:", min_value=0.0, max_value=120.0, value=0.5)
T = st.number_input("T:", min_value=36.0, max_value=40.0, value=37.0)
P = st.number_input("P:", min_value=50.0, max_value=150.0, value=90.0)
CRP = st.number_input("CRP:", min_value=1.0, max_value=400.0, value=100.0)
RDW_SD = st.number_input("RDW-SD:", min_value=30.0, max_value=80.0, value=50.0)

# 如果是二分类变量，这里设置选择框
#Carbapenems = st.selectbox("Chest pain type:", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

# Process inputs and make predictions
feature_values = [LDH, Mb, CK_MB, PCT, T, P,CRP,RDW_SD]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    threshold = 0.43  # 设置新的阈值
    predicted_proba = model.predict_proba(features)[0]  # # 获取第一个样本的预测概率（这是一个概率数组）
    predicted_class = 1 if predicted_proba[1] >= threshold else 0  #根据类别 1 的概率是否大于阈值调整类别

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, this patient have a high risk of initial treatment failure. "
            f"The model predicts that the probability of experiencing initial treatment failure is {probability:.1f}%. "
            f"While this is just an estimate, it suggests that you may be at significant risk. "
            f"We recommend closely monitoring this patient's condition and carefully selecting the initial antibiotic therapy. "
            f"It is also essential to conduct follow-up assessments to ensure the effectiveness of the initial treatment. "
        )
    else:
        advice = (
            f"According to our model, you have a low risk of initial treatment failure. "
            f"The model predicts that your probability of experiencing initial treatment effective is {probability:.1f}%. "
            f"While this is just an estimate, it suggests that you are at a relatively low risk. "
            f"Based on this, no immediate concerns are raisebd, but it is still important to monitor your condition and ensure that the chosen initial antibiotic therapy is effective. "
            f"Regular follow-up assessments will help confirm that the treatment is proceeding as expected. "
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # 绘制 SHAP force plot，确保传递 base_value 为第一个参数  如果是RF模型则用explainer.expected_value[1], shap_values[1]
    shap.plots.force(explainer.expected_value, shap_values, pd.DataFrame([feature_values], columns=feature_names),
                     matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")

#确保在命令行中运行 Streamlit 应用程序，使用以下命令：
#按 Win + R 打开运行对话框，输入 cmd 并按回车，打开命令提示符。
# 在python的终端 输入命令：streamlit run "D:\project\Streamlit在线应用程序开发.py"
