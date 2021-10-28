# Realtime-OCR-Text-Detection
Realtime OCR and Text Detection from Videos and Webcam, and Send detected text to serial port

# REQUIREMENTS
- EAST (An Efficient and Accurate Scene Text Detector) model for Text Detection It is a scene text detector that directly produces word or line level predictions from full images with a single neural network.
   Link to the paper: https://arxiv.org/pdf/1704.03155.pdf

- Pytesseract for Recognition The output bounding box of the detected text is localized and given as input to the pytesseract tool. Pytesseract is a powerful OCR tool used to recognize text.
   Download and Install PyTesseract from this repository or this link https://digi.bib.uni-mannheim.de/tesseract/ and Install it in your desired location.

- Download the east detector model(frozen_east_text_detection.pb) and keep in same director 


Instructions to run: 
Required libraries are OpenCV, pytesseract, imutils
Command to run : python3 text_extraction_from_video.py

# OUTPUT
![out](https://user-images.githubusercontent.com/91828519/139328756-1ffe581f-a9cd-408a-b4b5-777b2e2a861b.png)
