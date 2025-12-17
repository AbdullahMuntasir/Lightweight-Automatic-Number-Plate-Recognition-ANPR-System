# @title SYNDICATE_ANPR_V1
# Step 1: Import libraries ----
!pip install easyocr opencv-python matplotlib numpy
import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Step 2: Upload image ----
print("Please upload an image containing a vehicle number plate:")
uploaded = files.upload()

for fn in uploaded.keys():
    image_path = fn
    print(f"Image uploaded: {image_path}")

# Step 3: Read and preprocess image ----
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Error reading image. Please upload a valid image file.")

# ---- Step 3.1: Original Image ----
plt.figure(figsize=(8,5))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

# ---- Step 3.2: Convert to Grayscale ----
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8,5))
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# ---- Step 3.3: Bilateral Filter (Denoising) ----
gray = cv2.bilateralFilter(gray, 11, 17, 17)
plt.figure(figsize=(8,5))
plt.imshow(gray, cmap='gray')
plt.title("After Bilateral Filter (Noise Reduction)")
plt.axis('off')
plt.show()

# ---- Step 3.4: Canny Edge Detection ----
edged = cv2.Canny(gray, 30, 200)
plt.figure(figsize=(8,5))
plt.imshow(edged, cmap='gray')
plt.title("Edges Detected (Canny)")
plt.axis('off')
plt.show()

# ---- Step 3.5: Contour Detection ----
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

image_contours = image.copy()
cv2.drawContours(image_contours, cnts, -1, (0, 255, 0), 2)
plt.figure(figsize=(8,5))
plt.imshow(cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB))
plt.title("Contours Detected")
plt.axis('off')
plt.show()

screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

# Step 4: Mask & Crop Number Plate ----
if screenCnt is not None:
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray[topx:bottomx+1, topy:bottomy+1]

    plt.figure(figsize=(8,5))
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Plate Region (Masked)")
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(8,5))
    plt.imshow(cropped, cmap='gray')
    plt.title("Cropped Number Plate (for OCR)")
    plt.axis('off')
    plt.show()
else:
    print("[INFO] No number plate contour detected. Using full image for OCR.")
    cropped = gray

# Step 5: OCR Recognition (EasyOCR) ----
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped)

text = " ".join([d[1] for d in result]).strip()
text = text.upper()
print(f"\n[INFO] Detected Text: {text}")


# Step 6: Visualize OCR Results ----
if screenCnt is not None:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    # Optionally add text label above it
    x, y, w, h = cv2.boundingRect(screenCnt)
    cv2.putText(image, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
else:
    print("[INFO] No contour found for visualization.")


plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Final Detected Number: {text}")
plt.axis('off')
plt.show()