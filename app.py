import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import timm

device = torch.device('cpu')
classes = ['fake', 'real']  # Change this if your ImageFolder mapping is different

# Load EfficientNet model
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
model.load_state_dict(torch.load('best_effnet_model.pth', map_location=device))
model.eval()

st.title("Real vs Fake Image Detector")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = tr(image).unsqueeze(0)
    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.nn.functional.softmax(out[0], dim=0)
        conf, pred = torch.max(probs, 0)
        st.write(f"Prediction: **{classes[pred]}**")
        st.write(f"Confidence: **{conf.item()*100:.2f}%**")