import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Define YourModel class
class Model(nn.Module):
    def __init__(self, in_channels=1):
        super(Model, self).__init__()
        # Define your object detection model here

    def forward(self, x):
        return self.model.forward(x)

# Load model
model = Model() # Replace with your model definition
model.load_state_dict(torch.load('Adam_00001_2_dataCampuran.pt', map_location=torch.device('cpu'))) # Replace with your model file name and adjust for the device used
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define UI
st.title('Deploying PyTorch Object Detection Model to Streamlit')

# Define prediction function
def object_detection(model, image_transforms, image_path):
    image = Image.open(image_path).convert('RGB')
    image = image_transforms(image).unsqueeze(0)

    output = model(image)
    # Process output to get detected objects and their coordinates
    # Modify this part according to your object detection model output format
    # For example, if using Faster R-CNN, you can parse the bounding box coordinates and class labels from the output
    
    return detected_objects

# Choose between camera or upload image
option = st.selectbox('Choose image source:', ('Camera', 'Upload Image'))

if option == 'Camera':
    st.write('Please take a picture')
    picture = st.camera_input("Take a picture")
    if picture is not None:
        detected_objects = object_detection(model, transform, picture)
        # Display image with bounding boxes
        # Modify this part to draw bounding boxes on the image
        st.image(picture, caption='Detected Objects')
        # Display detected objects and their coordinates
        st.write(detected_objects)
else:
    uploaded_file = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        detected_objects = object_detection(model, transform, uploaded_file)
        # Display image with bounding boxes
        # Modify this part to draw bounding boxes on the image
        st.image(uploaded_file, caption='Detected Objects')
        # Display detected objects and their coordinates
        st.write(detected_objects)
