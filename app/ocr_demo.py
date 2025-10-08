# Import the necessary library
from paddleocr import PaddleOCR

# Initialize the PaddleOCR instance. 
# It will automatically download and load the models on the first run.
ocr = PaddleOCR(
    use_angle_cls=True, # Enables text angle classification
    lang='en' # Set the language to English
)

# Define the path to your image (can be a local path or a URL)
img_path = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"

# Run the OCR prediction
result = ocr.predict(input=img_path)

# Print the results
print("OCR Results:")
for res in result:
    # The 'res' object contains bounding box coordinates and the recognized text
    res.print() 
    
    # You can also save the results to an image or JSON file
    # res.save_to_img("output")
    # res.save_to_json("output")