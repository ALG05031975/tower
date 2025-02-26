import os
import cv2
import streamlit as st
import pandas as pd
from ultralytics import YOLO
import numpy as np
from PIL import Image

def load_model():
    return YOLO("best04.pt")

def predict_and_annotate(image_path, image_name):
    model = load_model()  # Reload model for each image to prevent batch carryover issues
    
    results = model(image_path, conf=0.5)[0]  # Use file path instead of NumPy array for consistency
    
    if len(results.boxes) == 0:
        return None, None, None  
    
    annotated_img = results.plot()
    
    bbox_data = []
    csv_data = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x_min, y_min, x_max, y_max = map(int, box.tolist())
        class_id = int(cls)  
        class_name = model.names[class_id]  
        bbox_data.append(f"{image_name}, {class_id}, {class_name}, {x_min}, {y_min}, {x_max}, {y_max}\n")
        csv_data.append([image_name, class_id, class_name, x_min, y_min, x_max, y_max])
    
    return annotated_img, bbox_data, csv_data

def main():
    st.title("Antenna Detection - Single & Batch Processing")
    
    mode = st.radio("Choose mode:", ["Single Image", "Process Multiple Images"])
    
    all_bbox_data = []
    all_csv_data = []
    
    if mode == "Single Image":
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            img_path = f"temp_{uploaded_file.name}"
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.image(img_path, caption="Uploaded Image")
            st.write("Detecting objects...")
            
            annotated_img, bbox_data, csv_data = predict_and_annotate(img_path, uploaded_file.name)
            
            if annotated_img is not None:
                annotated_pil = Image.fromarray(annotated_img)
                st.image(annotated_pil, caption="Annotated Image")
                
                img_output_path = f"annotated_{uploaded_file.name}"
                annotated_pil.save(img_output_path)
                with open(img_output_path, "rb") as f:
                    st.download_button("Download Annotated Image", f, file_name=img_output_path, mime="image/png")
                
                bbox_file_path = f"{uploaded_file.name}_bbox.txt"
                with open(bbox_file_path, "w") as f:
                    f.writelines(bbox_data)
                with open(bbox_file_path, "rb") as f:
                    st.download_button("Download Bounding Box Data (TXT)", f, file_name=bbox_file_path, mime="text/plain")
                
                csv_file_path = f"{uploaded_file.name}_bbox.csv"
                df = pd.DataFrame(csv_data, columns=["Image Name", "Class ID", "Class Name", "X Min", "Y Min", "X Max", "Y Max"])
                df.to_csv(csv_file_path, index=False)
                with open(csv_file_path, "rb") as f:
                    st.download_button("Download Bounding Box Data (CSV)", f, file_name=csv_file_path, mime="text/csv")
            else:
                st.write("No objects detected in the image.")
    
    elif mode == "Process Multiple Images":
        uploaded_files = st.file_uploader("Upload multiple images...", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                img_path = f"temp_{uploaded_file.name}"
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.image(img_path, caption=f"Uploaded - {uploaded_file.name}")
                st.write(f"Detecting objects in {uploaded_file.name}...")
                
                annotated_img, bbox_data, csv_data = predict_and_annotate(img_path, uploaded_file.name)
                
                if annotated_img is not None:
                    annotated_pil = Image.fromarray(annotated_img)
                    st.image(annotated_pil, caption=f"Annotated - {uploaded_file.name}")
                    
                    img_output_path = f"annotated_{uploaded_file.name}"
                    annotated_pil.save(img_output_path)
                    with open(img_output_path, "rb") as f:
                        st.download_button(f"Download {uploaded_file.name}", f, file_name=img_output_path, mime="image/png")
                    
                    all_bbox_data.extend(bbox_data)
                    all_csv_data.extend(csv_data)
            
            if all_bbox_data:
                bbox_file_path = "batch_bbox_data.txt"
                with open(bbox_file_path, "w") as f:
                    f.writelines(all_bbox_data)
                with open(bbox_file_path, "rb") as f:
                    st.download_button("Download All Bounding Box Data (TXT)", f, file_name=bbox_file_path, mime="text/plain")
                
            if all_csv_data:
                csv_file_path = "batch_bbox_data.csv"
                df = pd.DataFrame(all_csv_data, columns=["Image Name", "Class ID", "Class Name", "X Min", "Y Min", "X Max", "Y Max"])
                df.to_csv(csv_file_path, index=False)
                with open(csv_file_path, "rb") as f:
                    st.download_button("Download All Bounding Box Data (CSV)", f, file_name=csv_file_path, mime="text/csv")

if __name__ == "__main__":
    main()
