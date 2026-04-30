import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "NLR": {"type": "numerical", "min": 0.0212765957446809, "max": 510.4, "default": 10},
    "LAR": {"type": "numerical", "min": 0.0090293453724605, "max": 0.624338624338624, "default": 0.1},
    "Cr": {"type": "numerical", "min": 18, "max": 936, "default": 50},
    "SII": {"type": "numerical", "min": 0.0319, "max": 39687.869704119, "default": 1000},
    "PTA": {"type": "numerical", "min": 14, "max": 136.5, "default": 40},
    "Cystatin_C": {"type": "numerical", "min": 0.12, "max": 8.76, "default": 2},
    "CRP": {"type": "numerical", "min": 3.2, "max": 401, "default": 40},
    "AAPR": {"type": "numerical", "min": 0.029479384446253, "max": 1.18974358974359, "default": 0.1},
}

# Streamlit 界面
st.title("Hepatic Encephalopathy Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])
feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(feature_df)[0]
    predicted_proba = model.predict_proba(feature_df)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of Hepatic Encephalopathy is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    st.pyplot(fig)

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)
    
    # 调试信息（可选）
    st.write("Debug Info:")
    st.write(f"Predicted class: {predicted_class}")
    st.write(f"SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        st.write(f"Number of SHAP arrays: {len(shap_values)}")
        for i, arr in enumerate(shap_values):
            st.write(f"Array {i} shape: {arr.shape}")
    else:
        st.write(f"SHAP values shape: {shap_values.shape}")

    try:
        # 处理不同的 SHAP 输出格式
        if isinstance(shap_values, list):
            # 多分类情况
            if predicted_class < len(shap_values):
                shap_array = shap_values[predicted_class]
                expected_value = explainer.expected_value[predicted_class]
            else:
                shap_array = shap_values[0]
                expected_value = explainer.expected_value[0]
        else:
            # 二分类或回归情况
            shap_array = shap_values
            expected_value = explainer.expected_value
        
        # 生成 SHAP 力图
        plt.figure()
        shap_plot = shap.force_plot(
            expected_value,
            shap_array[0],  # 取第一个样本的SHAP值
            feature_df.iloc[0],
            matplotlib=True,
            show=False
        )
        
        # 显示 SHAP 图
        st.subheader("SHAP Force Plot")
        st.pyplot(plt.gcf())
        plt.close()
        
    except Exception as e:
        st.error(f"Error generating SHAP plot: {str(e)}")
        
        # 备用方案：显示特征重要性
        st.subheader("Feature Importance (Fallback)")
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': list(feature_ranges.keys()),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['feature'], feature_importance['importance'])
            ax.set_xlabel('Feature Importance')
            ax.set_title(' Feature Importance for Hepatic Encephalopathy')
            st.pyplot(fig)