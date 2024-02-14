import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F

# Define YourModel class
class Model(nn.Module):
    def __init__(self, in_channels=1):
        super(Model, self).__init__()

        self.model = models.mobilenet_v2(pretrained=True)

        # Tambahkan layer dropout setelah layer convolution
        self.model.features[13].dropout = nn.Dropout(0.2)
        self.model.features[15].dropout = nn.Dropout(0.2)
        self.model.features[17].dropout = nn.Dropout(0.2)

        # Ganti layer fc dengan layer fc baru dan tambahkan layer dropout
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.last_channel, 2)
        )

    def forward(self, x):
        return self.model.forward(x)

# Load model
model = Model() # Ganti dengan definisi model Anda
model.load_state_dict(torch.load('D:\Streamlit_skripsi\Adamax_00001_3.pt', map_location=torch.device('cpu'))) # Ganti dengan nama file model Anda dan sesuaikan dengan perangkat yang digunakan
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tentukan perangkat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels_data ={
    0: "Layak",
    1: "Tidak Layak"
}

def image_pred(model, image_transforms, image_path, labels_data):
    image = Image.open(image_path).convert('RGB')

    resize_image = image.resize((128,128))

    image = image_transforms(image).float()
    image = image.unsqueeze(0).to(device)

    output = model(image)
    probs = F.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)

    return labels_data[classes.item()], round(conf.item(), 3), resize_image

# Define UI
st.title('Deploying PyTorch Model to Streamlit')

# Pilihan menggunakan kamera atau unggah gambar
option = st.selectbox('Pilih sumber gambar:', ('Kamera', 'Unggah Gambar'))

if option == 'Kamera':
    st.write('Silakan ambil gambar')
    picture = st.camera_input("Take a picture")
    if picture is not None:
        predicted_class, confidence, resized_image = image_pred(model, transform, picture, labels_data)
        st.image(resized_image)
        st.write('Predicted Class:', predicted_class)
        st.write('Confidence:', confidence)
        # Tampilkan progress bar untuk confidence
        progress_bar = st.progress(confidence)
        
        # Menampilkan notifikasi berdasarkan prediksi
        if predicted_class == "Layak":
            st.success("Selamat! Anda mendapatkan 1000 koin🪙 sebagai reward karena botol ini layak untuk dijadikan kerajinan!")
        else:
            st.warning("Botol ini tidak layak untuk dijadikan kerajinan, Anda mendapatkan 100 koin🪙.")
else:
    uploaded_file = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        predicted_class, confidence, resized_image = image_pred(model, transform, uploaded_file, labels_data)
        st.image(resized_image)
        st.write('Predicted Class:', predicted_class)
        
        st.write('Confidence:', confidence)
        # Tampilkan progress bar untuk confidence
        progress_bar = st.progress(confidence)
        
        # Menampilkan notifikasi berdasarkan prediksi
        if predicted_class == "Layak":
            st.success("Selamat! Anda mendapatkan 1000 koin🪙 sebagai reward karena botol ini layak untuk dijadikan kerajinan!")
        else:
            st.warning("Botol ini tidak layak untuk dijadikan kerajinan, Anda mendapatkan 100 koin🪙.")
