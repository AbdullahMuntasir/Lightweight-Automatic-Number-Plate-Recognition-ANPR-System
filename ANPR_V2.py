# @title SYNDICATE_ANPR_V2

# Step 1: Import libraries
import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import re

#RGB color and bold
def rgb_color_bold(r, g, b, text):
    return f'\033[1m\033[38;2;{r};{g};{b}m{text}\033[0m'

"""print(rgb_color_bold(0, 150, 0, "Medium Green (0, 150, 0)"))
print(rgb_color_bold(148, 0, 211, "Medium Purple (148, 0, 211)"))
print(rgb_color_bold(255, 110, 0, "Dark Orange (255, 140, 0)"))
print(rgb_color_bold(255, 0, 0, "Pure Red (255, 0, 0)"))
print(rgb_color_bold(0, 0, 255, "Pure Blue (0, 0, 255)"))"""

# ----------------- Helper functions ----------------- #

def preprocess_for_ocr(roi_gray, scale=3):
    """Resize + binarize plate ROI for better OCR, with visualization."""
    h, w = roi_gray.shape[:2]

    # --- Resize (Upsampling) ---
    resized = cv2.resize(roi_gray, (w * scale, h * scale),
                         interpolation=cv2.INTER_CUBIC)

    # --- Threshold (Binarization using Otsu) ---
    _, thresh = cv2.threshold(
        resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    plt.figure(figsize=(6,3))
    plt.imshow(thresh, cmap='gray')
    plt.title("Thresholded (Otsu Binarization)")
    plt.axis('off')
    plt.show()

    return thresh

def fix_z2_by_pattern(plate):
    """Apply Z <-> 2 correction for Indian-style plates."""
    p = list(plate)
    n = len(p)

    if n == 10:          # LLNNLLNNNN
        letter_pos = [0, 1, 4, 5]
        digit_pos  = [2, 3, 6, 7, 8, 9]
    elif n == 9:         # approx LLNNLNNNN
        letter_pos = [0, 1, 4]
        digit_pos  = [2, 3, 5, 6, 7, 8]
    else:
        return plate

    for i in letter_pos:
        if p[i] == '2':
            p[i] = 'Z'
    for i in digit_pos:
        if p[i] == 'Z':
            p[i] = '2'

    return ''.join(p)

def postprocess_plate_text(raw_text):
    """Clean OCR text and apply Indian-pattern correction if possible."""
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    print("[DEBUG] Cleaned OCR:", cleaned)

    patterns = [
        r"[A-Z]{2}\d{2}[A-Z]{2}\d{4}",
        r"[A-Z]{2}\d{2}[A-Z]{1}\d{4}",
        r"[A-Z]{2}\d{2}\d{4}",
    ]

    for pat in patterns:
        m = re.search(pat, cleaned)
        if m:
            candidate = m.group(0)
            fixed = fix_z2_by_pattern(candidate)
            print("[DEBUG] Indian pattern:", candidate, "->", fixed)
            return fixed

    # Non-Indian / unknown format
    return cleaned

def select_best_text_box(results, img_shape):
    """
    From EasyOCR results, pick the box that looks most like a plate.
    Heuristics:
      - in lower half of image (avoid windshield)
      - long & thin (aspect ratio)
      - reasonable area
      - decent confidence
    """
    H, W = img_shape[:2]
    img_area = H * W

    best = None
    best_score = -1

    for (bbox, text, conf) in results:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        w_box = x_max - x_min
        h_box = y_max - y_min
        if w_box <= 0 or h_box <= 0:
            continue

        area = w_box * h_box
        aspect = w_box / float(h_box)
        cy = (y_min + y_max) / 2.0

        # Area filter
        if area < 0.0003 * img_area or area > 0.2 * img_area:
            continue

        # Aspect ratio filter (long & thin)
        if aspect < 2.0 or aspect > 10.0:
            continue

        # Only look at lower half
        if cy < 0.5 * H:
            continue

        # Prefer around 80% height (near bumper)
        pos_score = 1.0 - abs(cy / H - 0.8)
        pos_score = max(0.0, pos_score)

        # Reasonable text length
        clen = len(re.sub(r'[^A-Z0-9]', '', text.upper()))
        if clen < 3 or clen > 12:
            continue

        score = conf + 0.5 * pos_score

        if score > best_score:
            best_score = score
            best = (int(x_min), int(y_min), int(x_max), int(y_max), text, conf)

    return best

# ----------------- Main program ----------------- #

# Step 2: Upload image
print(rgb_color_bold(148, 0, 211,"[INFO] Please upload an image containing a vehicle number plate:"))
uploaded = files.upload()

for fn in uploaded.keys():
    image_path = fn
    #print(rgb_color_bold(0, 150, 0, f"[INFO] Image uploaded: {image_path}"))
    print(rgb_color_bold(255, 110, 0, f"[INFO] Image uploaded: {image_path}"))

# Step 3: Read image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(rgb_color_bold(255, 0, 0,"[INFO] Error reading image. Please upload a valid image file."))


# Optional: upscale small images
H, W = image.shape[:2]
if max(H, W) < 900:
    scale = 900.0 / max(H, W)
    image = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)
    H, W = image.shape[:2]
    print(f"[INFO] Upscaled image by factor {scale:.2f}")

# Step 4: First OCR pass – detect all text boxes on full image
reader = easyocr.Reader(['en'])

import threading, time, sys

running = True

def animate_dots():
    dots = ["", ".", "..", "..."]
    i = 0
    while running:
        sys.stdout.write(f"\r[INFO] Running EasyOCR on full image to get text boxes{dots[i]}")
        sys.stdout.flush()
        i = (i + 1) % 4
        time.sleep(0.4)

# Start animation
t = threading.Thread(target=animate_dots)
t.start()

# Run OCR (blocking)
results = reader.readtext(image)

# Stop animation
running = False
t.join()

print("\r[INFO] Running EasyOCR on full image to get text boxes... Done ✔️")



print(f"[INFO] Number of text boxes found: {len(results)}")

# Step 5: Choose the box most likely to be the plate
best_box = select_best_text_box(results, image.shape)

if best_box is None:
    print(rgb_color_bold(255, 0, 0, "[INFO] No clear plate-like box found. Using highest-confidence text as fallback."))
    if results:
        fallback_text = max(results, key=lambda x: x[2])[1]
        final_text = postprocess_plate_text(fallback_text)
    else:
        final_text = ""
    cropped = None
else:
    x1, y1, x2, y2, raw_text, conf = best_box
    print(f"[INFO] Selected candidate: '{raw_text}' (conf={conf:.2f})")

    # Step 6: Crop the detected plate region
    cropped_bgr = image[y1:y2, x1:x2]
    gray_cropped = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(6, 3))
    plt.imshow(gray_cropped, cmap='gray')
    plt.title("Detected Number Plate - Grayscaled")
    plt.axis('off')
    plt.show()

    # Step 7: Second OCR pass on cropped ROI (with allowlist)
    cropped_proc = preprocess_for_ocr(gray_cropped)
    result_roi = reader.readtext(
        cropped_proc, detail=1,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    if result_roi:
        text_roi = "".join([r[1] for r in result_roi])
    else:
        text_roi = raw_text  # fallback

    print("[INFO] Raw text from ROI:", text_roi)
    final_text = postprocess_plate_text(text_roi)
    cropped = cropped_bgr

#print(f"\n Final Detected Text: {final_text}")
print(rgb_color_bold(0, 150, 0, f"\n[INFO] Final Detected Text: {final_text}"))

# Step 8: Visualize detections on original image
annotated = image.copy()
display_text = final_text if final_text else "Plate not recognized"

if best_box is not None:
    x1, y1, x2, y2, _, _ = best_box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(annotated, display_text, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
else:
    cv2.putText(annotated, display_text, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 2)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.title(f"Final Detected Number: {display_text}")
plt.axis('off')
plt.show()