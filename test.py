import hailo_platform.pyhailort as pyhailort
import numpy as np
from PIL import Image

# Path to your HEF file
hef_file_path = "ArcFaceMobileFaceNet"

# Function to preprocess the image
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_shape[1], input_shape[0]))  # Resize to model input dimensions
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_array.astype(np.float32)

# Load the HEF file
with pyhailort.Device() as device:
    with pyhailort.VdmaConfigManager(device, hef_file_path) as vdma_manager:
        # Get input/output sizes
        input_vstream_info = vdma_manager.get_input_vstreams()[0]
        output_vstream_info = vdma_manager.get_output_vstreams()[0]
        
        input_shape = input_vstream_info.shape
        output_shape = output_vstream_info.shape
        
        # Example: Prepare input data
        image_path = "faces/person1/img0.jpg"  # Replace with your test image
        input_data = preprocess_image(image_path, input_shape)
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

        # Run inference
        with vdma_manager.create_vstreams() as vstreams:
            results = vstreams.input[0].write(input_data)
            output_data = vstreams.output[0].read()
            print("Inference results:", output_data)
