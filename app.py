import matplotlib.pyplot as plt
import streamlit as st
import torchvision
from icedata.utils import load_model_weights_from_url
from icevision.all import *
from PIL import Image

CLASS_MAP = ClassMap({"background": 0, "syringe": 1, "glove": 2, "mask": 3})
WEIGHTS_URL = "https://github.com/benihime91/Covid-Waste-Detection/releases/download/v1.0/effecientdnet-b0-ver0.01-3VNfX.pth"
SIZE = 512
TRANSFORMS_LIST = [*tfms.A.resize_and_pad(SIZE), tfms.A.Normalize()]
TRANSFORMS = tfms.A.Adapter(TRANSFORMS_LIST)
MODEL_TYPE = models.ross.efficientdet

# TODO Way to load model with @st.cache so it doesn't take a long time each time
@st.cache()
def load_model():
    # fmt: off
    backbone = MODEL_TYPE.backbones.d0(pretrained=False)
    infer_model = efficientdet.model(backbone=backbone, num_classes=len(CLASS_MAP), img_size=SIZE)
    load_model_weights_from_url(infer_model, WEIGHTS_URL, map_location=torch.device("cpu"))
    return infer_model
    # fmt: on


# Inference function - TODO this could probably be improved ...
def predict_bboxes(image, threshold=0.5):
    # Create predictor
    with st.spinner("🏃‍♂️ Getting the latest model weights ..."):
        predictor = load_model()
    st.success("🚀 Model Weights Loaded Successfully !")
    # Convert PIL image to array
    image = [np.asarray(image)]
    infer_ds = Dataset.from_images(images=image, tfm=TRANSFORMS, class_map=CLASS_MAP)
    with st.spinner("🏃‍♂️ Doing the Math 🤓 ..."):
        preds = MODEL_TYPE.predict(predictor, infer_ds, detection_threshold=threshold)
    st.success("🚀 Predictions Generated !")
    preds = draw_pred(preds[0], denormalize_fn=denormalize_imagenet)
    preds = np.uint8(preds)
    preds = Image.fromarray(preds)
    return preds


def _processed_image(image):
    """
    Display Processed Image to the User for Sanity Check
    """
    # Convert PIL image to array
    image = [np.asarray(image)]
    infer_ds = Dataset.from_images(images=image, tfm=TRANSFORMS, class_map=CLASS_MAP)
    infer_dl = MODEL_TYPE.infer_dl(infer_ds, batch_size=1)
    x, sample = first(infer_dl)
    grid = torchvision.utils.make_grid(x[0], normalize=True)
    grid = torchvision.transforms.ToPILImage()(grid)
    return grid


# fmt: off
def main():
    title = """
    # Covid 19 Waste Detector 👁  

    This application tries to detect waste medical generated during this COVID 19 Pandemic.  
    To be more specific this application detect the presence of the following instances given an image.
    - Face Masks
    - Medical Syringes
    - Medical Hand Gloves
    """
    st.markdown(title)

    instructions = """
    ## How does it work?  
    Add an image and a machine learning learning model will look at it and find the instances 
    like the example shown below:
    """
    st.sidebar.write(instructions)
    st.sidebar.image(Image.open("TEST_IMAGE.jpeg"), use_column_width=True)

    thres = st.sidebar.slider(label="Detection threshold", min_value=0.1, max_value=1.0, value=0.4)

    note = """
    *Tip*:  
    You can adjust this sidebar and modify the detection score thresholds. What this essentially 
    mean that the detections whose score probabilities are lesser than the threshold will not be
    selected."""
    st.sidebar.write(note)

    instructions = """
    The image you select or upload will be fed through the Object Detection Network in real-time and
    the output will be displayed to the screen.  
    
    *Note*: The model was originally trained on images of 512 pixels. So for best results use images
    with dimensions greater than or equal to 512 pixels. If image of other dimensions are loaded, we will
    preprocess the images and get them to the correct size either by cropping or padding."""
    st.write(instructions)

    uploaded_image = st.file_uploader("Upload An Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image = _processed_image(image)
        st.title("Here is the image you've selected")
        st.image(image, caption="Processed Image", use_column_width=False)
        predictions = predict_bboxes(image, threshold=thres)
        st.title("Model Predictions")
        st.image(predictions, use_column_width=False)

if __name__ == "__main__":
    # run the app
    main()

