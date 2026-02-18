
import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import json
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Ayurvedic Diet", layout="wide")

@st.cache_resource
def load_model():
    device = torch.device("cpu")  # Cloud uses CPU
    with open("labels.json", "r") as f:
        labels = json.load(f)
    model = torch.load("food_model_complete.pth", map_location=device)
    model.eval()
    return model, labels, device

st.title("🌿 Ayurvedic Diet Planner PS6")
st.markdown("**Upload food image for dosha-based analysis**")

try:
    model, labels, device = load_model()
    st.success("✅ Model loaded (86% accuracy)!")
except:
    st.error("Upload model files first")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Choose food image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)
    
    if st.button("🔍 Analyze Food (Ayurvedic)", type="primary"):
        with st.spinner("Analyzing..."):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img)
                probs = torch.nn.functional.softmax(outputs[0], 0)
                top3 = probs.topk(3)
            
            st.subheader("🍽️ Top 3 Predictions")
            for i, (prob, idx) in enumerate(zip(top3.values, top3.indices)):
                food = labels[idx].replace("_", " ").title()
                st.write(f"{i+1}. **{food}** - {prob.item()*100:.1f}% confidence")

# Simple dosha guide
with st.expander("📋 Quick Dosha Guide"):
    st.write("- **Vata**: Warm, moist, grounding foods")
    st.write("- **Pitta**: Cooling, sweet, bitter foods") 
    st.write("- **Kapha**: Light, warm, spicy foods")
