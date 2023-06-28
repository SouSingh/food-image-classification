import streamlit as st
from PIL import Image
from model import preprocess

def process_image(image):
    x = preprocess(img_path=image)
    return x

def main():
    st.title("Image Processing App")
    st.write("Upload an image and see the processed output.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        # Process the image
        prediction = process_image(uploaded_file)
        st.write("Predication:", prediction)

if __name__ == "__main__":
    main()
