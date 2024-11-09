import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import matplotlib.pyplot as plt

# Expanded dictionary containing treatments for various diseases
disease_treatments = {
    "Grape with Black Rot": "Prune affected areas, avoid water on leaves, use fungicide if severe.",
    "Potato with Early Blight": "Apply fungicides, avoid overhead watering, rotate crops yearly.",
    "Tomato with Early Blight": "Remove infected leaves, use copper-based fungicide, maintain good airflow.",
    "Apple with Scab": "Remove fallen leaves, prune trees, apply fungicide in early spring.",
    "Wheat with Leaf Rust": "Apply resistant varieties, use fungicides, remove weeds.",
    "Cucumber with Downy Mildew": "Use resistant varieties, ensure good air circulation, apply fungicide.",
    "Rose with Powdery Mildew": "Use sulfur or potassium bicarbonate sprays, prune affected areas, avoid overhead watering.",
    "Strawberry with Gray Mold": "Remove infected fruits, improve ventilation, avoid wetting the fruit when watering.",
    "Peach with Leaf Curl": "Apply a fungicide in late fall or early spring, remove affected leaves.",
    "Banana with Panama Disease": "Use disease-resistant varieties, ensure soil drainage, avoid overwatering.",
    "Tomato with Septoria Leaf Spot": "Use resistant varieties, remove infected leaves, apply fungicide.",
    "Corn with Smut": "Remove infected ears, use disease-free seed, rotate crops.",
    "Carrot with Root Rot": "Ensure well-draining soil, avoid excessive watering, use crop rotation.",
    "Onion with Downy Mildew": "Use fungicides, ensure adequate spacing, avoid overhead watering.",
    "Potato with Late Blight": "Apply copper-based fungicides, remove affected foliage, practice crop rotation.",
    "Citrus with Greening Disease": "Remove infected trees, control leafhopper population, plant disease-free trees.",
    "Lettuce with Downy Mildew": "Ensure good air circulation, avoid overhead watering, apply fungicides.",
    "Pepper with Bacterial Spot": "Use resistant varieties, apply copper-based bactericides, practice crop rotation.",
    "Eggplant with Verticillium Wilt": "Use resistant varieties, solarize soil before planting, avoid soil disturbance.",
    "Cotton with Boll Rot": "Improve drainage, remove infected bolls, apply fungicides if necessary.",
    "Soybean with Soybean Rust": "Use fungicides, rotate crops, use resistant varieties if available.",
    "Rice with Sheath Blight": "Reduce nitrogen application, maintain proper water levels, apply fungicides.",
    "Sunflower with Downy Mildew": "Use resistant varieties, avoid waterlogging, apply fungicides.",
    "Barley with Net Blotch": "Use resistant varieties, remove crop residues, apply fungicides.",
    "Oat with Crown Rust": "Use resistant varieties, apply fungicides, avoid high nitrogen levels.",
    "Sugarcane with Red Rot": "Use disease-free cuttings, control weeds, apply fungicides if necessary.",
    "Pine with Pine Wilt": "Remove and destroy infected trees, control beetle population, avoid planting susceptible species.",
    "Avocado with Anthracnose": "Prune infected branches, use copper-based fungicides, avoid wet foliage.",
    "Papaya with Papaya Ringspot Virus": "Use virus-resistant varieties, remove infected plants, control aphid population.",
    "Mango with Powdery Mildew": "Use sulfur-based fungicides, remove affected parts, avoid overhead watering.",
    "Peanut with Leaf Spot": "Use resistant varieties, apply fungicides, rotate crops to reduce infection risk.",
    "Chili with Anthracnose": "Apply copper fungicides, remove infected fruits, avoid overhead irrigation.",
    "Garlic with White Rot": "Remove infected plants, improve soil drainage, practice crop rotation."
    # You can add more diseases and treatments here
}

# Streamlit title and description
st.title("Plant Disease Detection")
st.write("Upload an image of a plant leaf, and the app will detect the disease and suggest a treatment.")

# File upload option
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL and display it
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Initialize the feature extractor and model
    extractor = AutoFeatureExtractor.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
    model = AutoModelForImageClassification.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")

    # Preprocess the image
    inputs = extractor(images=image, return_tensors="pt")

    # Get the model's raw prediction (logits)
    outputs = model(**inputs)
    logits = outputs.logits

    # Temperature scaling for logits (lowering temperature to adjust confidence)
    temperature = 0.5
    logits = logits / temperature

    # Convert logits to probabilities using softmax
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(logits)

    # Get the top prediction
    top_k = 1
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    top_probs = top_probs[0].tolist()
    top_indices = top_indices[0].tolist()

    # Define a confidence threshold
    confidence_threshold = 0.7
    class_labels = model.config.id2label
    predicted_disease = "Unknown Disease"
    predicted_confidence = top_probs[0]

    if predicted_confidence >= confidence_threshold:
        predicted_disease = class_labels.get(top_indices[0], "Unknown Disease")

    # Display prediction and treatment
    st.write(f"**Predicted Disease:** {predicted_disease}")
    st.write(f"**Confidence:** {predicted_confidence:.4f}")

    # Provide treatment if disease is known
    treatment = disease_treatments.get(predicted_disease, "No treatment information available.")
    st.write(f"**Suggested Treatment:** {treatment}")
