import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
import os
import textwrap
from reportlab.lib.utils import ImageReader
from openai import OpenAI  # For openai>=1.0.0

# === Load environment variables ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# === Load model and scaler ===
model = joblib.load("Farm_Irrigation_XGB.pkl")
scaler = joblib.load("scaler.pkl")
history_file = "irrigation_history.csv"

# === Page setup ===
st.set_page_config(layout="wide")
st.title("🌾 Smart Farm Irrigation Predictor")
st.markdown("Enter sensor readings to predict irrigation needs for:\n"
            "-  Rice Field\n-  Wheat Crop\n-  Vegetable Plot")

# === Sensor Features ===
feature_names = [
    'soil_moisture_top', 'soil_moisture_middle', 'soil_moisture_bottom',
    'temperature_surface', 'temperature_subsoil',
    'humidity_surface', 'humidity_subsoil', 'ph_level',
    'electrical_conductivity', 'rainfall_mm',
    'wind_speed', 'solar_radiation', 'water_pressure',
    'evapotranspiration', 'leaf_wetness', 'nitrogen_content',
    'phosphorus_content', 'potassium_content',
    'soil_temperature', 'canopy_temperature'
]

# === Session State Initialization ===
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = []
if 'result' not in st.session_state:
    st.session_state.result = []
if 'explanation' not in st.session_state:
    st.session_state.explanation = ""
if 'download_triggered' not in st.session_state:
    st.session_state.download_triggered = False

# === Sensor Input ===
st.subheader(" Sensor Input")
user_input = []

with st.form("sensor_form"):
    cols = st.columns(2)
    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            val = st.number_input(
                f"**{feature.replace('_', ' ').title()}**",
                step=0.01, format="%.2f", placeholder="Enter value"
            )
            user_input.append(val)
    if st.form_submit_button(" Predict Irrigation Zones"):
        st.session_state.user_input = user_input
        input_scaled = scaler.transform([user_input])
        prediction = model.predict(input_scaled)[0]
        zones = ['Rice Field', 'Wheat Crop', 'Vegetable Plot']
        result = [zones[i] for i, pred in enumerate(prediction) if pred == 1]
        st.session_state.result = result
        st.session_state.submitted = True

        with st.spinner(" Generating AI explanation..."):
            try:
                sensor_context = ", ".join([
                    f"{k.replace('_',' ')}: {v:.2f}"
                    for k, v in zip(feature_names, user_input)
                ])
                prompt = (
                    f"The following sensor data was collected from a farm:\n"
                    f"{sensor_context}.\n"
                    f"Based on this, the system recommended irrigation for: {', '.join(result) if result else 'None'}.\n"
                    f"Explain this recommendation in simple language that a farmer can understand."
                )

                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": "You are an agriculture assistant. Explain irrigation decisions simply."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )

                explanation = response.choices[0].message.content.strip()
                st.session_state.explanation = explanation

            except Exception as e:
                st.error(f" Groq API Error: {e}")

# === Show Results if Submitted ===
if st.session_state.submitted:
    user_input = st.session_state.user_input
    result = st.session_state.result
    explanation = st.session_state.explanation

    if result:
        st.success(f" Recommended Irrigation: {', '.join(result)}")
    else:
        st.info(" No irrigation needed at this time.")

    if explanation:
        st.subheader(" AI Explanation for Recommendation (via Groq)")
        st.markdown(explanation)

    # === Save to History ===
    history_row = dict(zip(feature_names, user_input))
    history_row["Prediction"] = ", ".join(result) if result else "None"
    try:
        df_history = pd.read_csv(history_file)
    except FileNotFoundError:
        df_history = pd.DataFrame(columns=feature_names + ["Prediction"])
    df_history = pd.concat([df_history, pd.DataFrame([history_row])], ignore_index=True)
    df_history.to_csv(history_file, index=False)

    # === Bar Graph ===
    st.divider()
    st.subheader(" Sensor Input Bar Graph")
    sensor_df = pd.DataFrame({'Sensor': feature_names, 'Value': user_input})
    bar_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(user_input)))
    fig_bar, ax_bar = plt.subplots(figsize=(12, 5))
    ax_bar.bar(sensor_df['Sensor'], sensor_df['Value'], color=bar_colors)
    ax_bar.set_title("Sensor Input Values")
    ax_bar.set_ylabel("Value")
    ax_bar.tick_params(axis='x', rotation=90)
    st.pyplot(fig_bar)

    # === Pie Chart ===
    st.subheader(" Sensor Value Distribution")
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    ax_pie.pie(user_input, labels=feature_names, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20c.colors)
    ax_pie.axis('equal')
    st.pyplot(fig_pie)

    # === Feature Importance ===
    st.subheader(" Feature Importance (from Model)")
    importances = [est.feature_importances_ for est in model.estimators_]
    avg_importance = np.mean(importances, axis=0)
    importance_df = pd.DataFrame({
        "Sensor": feature_names,
        "Importance": avg_importance
    }).sort_values(by="Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    ax_imp.barh(importance_df["Sensor"], importance_df["Importance"], color='skyblue')
    ax_imp.set_title("Which Sensors Influenced the Prediction?")
    ax_imp.invert_yaxis()
    st.pyplot(fig_imp)

    # === Historical Comparison ===
    st.subheader(" Comparison with Historical Sensor Averages")
    if len(df_history) >= 5:
        avg_df = df_history[feature_names].astype(float).mean()
        comparison_df = pd.DataFrame({
            "Sensor": feature_names,
            "Current Value": user_input,
            "Historical Avg": avg_df.values
        })
        fig_cmp, ax_cmp = plt.subplots(figsize=(12, 5))
        x = np.arange(len(feature_names))
        width = 0.35
        ax_cmp.bar(x - width/2, comparison_df["Current Value"], width, label='Current')
        ax_cmp.bar(x + width/2, comparison_df["Historical Avg"], width, label='Historical Avg')
        ax_cmp.set_xticks(x)
        ax_cmp.set_xticklabels(feature_names, rotation=90)
        ax_cmp.legend()
        st.pyplot(fig_cmp)
    else:
        st.info("ℹ Add at least 5 history records for meaningful comparison.")

    # === History Table ===
    st.subheader(" Past Prediction History")
    st.dataframe(df_history.tail(10).sort_index(ascending=False), use_container_width=True)

    # === PDF Export ===
    st.subheader("Export Report as PDF")
    if st.button("Download Report"):
        st.session_state.download_triggered = True

    if st.session_state.download_triggered:
        img_bufs = []
        figures_to_save = [fig_bar, fig_pie, fig_imp]
        if 'fig_cmp' in locals():
            figures_to_save.append(fig_cmp)
        
        image_bytes_list = []

        for fig in figures_to_save:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            image_bytes_list.append(buf.read())

        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        margin = 50
        
        y = height - margin

        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, y, "Smart Farm Irrigation Report")
        y-= 30

        c.setFont("Helvetica", 12)
        c.drawString(margin, y, f"Recommended Irrigation Zone(s): {', '.join(result)}")
        y -= 20

        if explanation:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, "AI Explanation:")
            y -= 18
            c.setFont("Helvetica", 10)
            for line in explanation.split('\n'):
                wrapped=textwrap.wrap(line,90)
                for wrap_line in wrapped:
                    c.drawString(margin, y, wrap_line)
                    y-= 14
                    if y < 100:
                        c.showPage()
                        y= height - margin
            y -= 10

        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Sensor Inputs:")
        y -= 18
        c.setFont("Helvetica", 10)
        
        
        sensor_lines = []
        for feat, val in zip(feature_names, user_input):
            clean_feat = feat.replace('_', ' ').title()
            sensor_lines.append(f"{clean_feat}: {val:.2f}")
        
        line_height = 14
        column_spacing = 260
        x_left = margin
        x_right = margin + column_spacing
        y_start = y
        y_pos = y_start

        for i in range(0, len(sensor_lines), 2):
            left = sensor_lines[i]
            right = sensor_lines[i+1] if i+1 < len(sensor_lines) else ""

            c.drawString(x_left, y_pos, left)
            c.drawString(x_right, y_pos, right)
 
            y_pos -= line_height
            if y_pos < 100:
                c.showPage()
                y_pos = height - margin
                c.setFont("Helvetica", 10) 




        for i in range(0, len(image_bytes_list), 2):
            c.showPage()
            images = image_bytes_list[i:i+2]
            positions = [(margin, height / 2 + 20), (margin, margin)]

            for img_data, (x, y_pos) in zip(images, positions):
                img = Image.open(io.BytesIO(img_data))
                img_width, img_height = img.size

                max_width = width - 2 * margin
                max_height = (height / 2) - 40
                scale = min(max_width / img_width, max_height / img_height)

                new_width = img_width * scale
                new_height = img_height * scale

                x_centered = (width - new_width) / 2
                img_reader = ImageReader(io.BytesIO(img_data))
                c.drawImage(img_reader, x_centered, y_pos, new_width, new_height)





        c.save()
        pdf_buffer.seek(0)

        st.download_button(
            label="Click to Download PDF Report",
            data=pdf_buffer,
            file_name="irrigation_report.pdf",
            mime="application/pdf"
        )
