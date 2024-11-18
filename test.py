# https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8/HAILO8_face_recognition.rst
import os
import cv2
import numpy as np
import hailo_platform as hp

# Define paths
hef_file_path = "ArcFaceMobileFaceNet.hef"
faces_folder_path = "faces"

# Preprocessing function
def preprocess_image(image_path, target_size=(112, 112)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image at {image_path}")
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Load and preprocess all face images in the folder
def load_faces(folder_path):
    face_images = []
    file_names = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            preprocessed_image = preprocess_image(file_path)
            face_images.append(preprocessed_image)
            file_names.append(file_name)
    return np.vstack(face_images), file_names

# Run inference on the preprocessed images
def run_inference(input_data):
    with hp.Hef(hef_file_path) as hef:
        with hp.Device() as device:
            network_group = hef.configure(device)
            with network_group.activate() as runner:
                input_vstream = runner.get_input_vstreams()[0]
                output_vstream = runner.get_output_vstreams()[0]

                # Send input data to the device
                runner.send(input_vstream.name, input_data)

                # Retrieve the inference results
                output = runner.receive(output_vstream.name)
                return output

# Main script
if __name__ == "__main__":
    # Load and preprocess images
    face_data, file_names = load_faces(faces_folder_path)
    print(f"Loaded {len(file_names)} images for inference.")

    # Run inference
    embeddings = run_inference(face_data)

    # Postprocess and display results
    for i, file_name in enumerate(file_names):
        print(f"Embedding for {file_name}: {embeddings[i]}")
