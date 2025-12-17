# @title SYNDICATE_ANPR_V3
import cv2, re
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# ---------- Basic text helpers ----------

def clean_text(t: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', t.upper())

def text_is_valid_plate(s: str) -> bool:
    if not (5 <= len(s) <= 12):
        return False
    return any(c.isalpha() for c in s) and any(c.isdigit() for c in s)

# ---------- Refined plate correction ----------

def fix_plate_layout(plate: str) -> str:
    p = list(plate)
    n = len(p)
    if n not in (9, 10):
        return plate

    if n == 10:              # LLNNLLNNNN
        letter_pos = [0, 1, 4, 5]
        digit_pos  = [2, 3, 6, 7, 8, 9]
    else:                    # 9 -> LLNNLNNNN
        letter_pos = [0, 1, 4]
        digit_pos  = [2, 3, 5, 6, 7, 8]

    map_digit = {           # used when we EXPECT digits
        'O':'0',
        'I':'1', 'L':'1',
        'Z':'2',
        'S':'5',
        'B':'8',
        'G':'6',
    }
    map_letter = {          # used when we EXPECT letters
        '0':'O',
        '1':'I',
        '2':'Z',
        '5':'S',
        '6':'G',
        '8':'B',
    }

    for i in digit_pos:
        if 0 <= i < n and p[i] in map_digit:
            p[i] = map_digit[p[i]]

    for i in letter_pos:
        if 0 <= i < n and p[i] in map_letter:
            p[i] = map_letter[p[i]]

    return ''.join(p)

def postprocess_plate_text(cleaned: str) -> str:
    if len(cleaned) in (9, 10) and cleaned[:2].isalpha() \
       and sum(c.isdigit() for c in cleaned) >= 4:
        fixed = fix_plate_layout(cleaned)
        print("[DEBUG] Refined layout:", cleaned, "->", fixed)
        return fixed
    return cleaned

# ---------- Image preprocessing ----------

def enhance_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.filter2D(g, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
    return g

def preprocess_for_ocr(roi_gray, scale=3):
    roi = cv2.bilateralFilter(roi_gray, 7, 75, 75)
    roi = enhance_gray(roi)
    h, w = roi.shape[:2]
    roi = cv2.resize(roi, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def expand_roi(image, x1, y1, x2, y2, expand_ratio=0.40):
    H, W = image.shape[:2]
    w = x2 - x1
    e = int(w * expand_ratio)
    x1n = max(0, x1 - e)
    x2n = min(W, x2 + e)
    y1n = max(0, y1 - 5)
    y2n = min(H, y2 + 5)
    return x1n, y1n, x2n, y2n

# ---------- Detection: OCR-box based ----------

def select_best_text_box(results, img_shape, verbose=True):
    H, W = img_shape[:2]
    img_area = H * W
    best, best_score = None, -1

    for (bbox, text, conf) in results:
        xs = [pt[0] for pt in bbox]; ys = [pt[1] for pt in bbox]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w_box, h_box = x_max - x_min, y_max - y_min
        if w_box <= 0 or h_box <= 0:
            continue

        area = w_box * h_box
        aspect = w_box / float(h_box)
        cy = (y_min + y_max) / 2.0
        cleaned = clean_text(text)

        if not text_is_valid_plate(cleaned):
            if verbose: print(f"[OCR BOX] ({int(x_min)},{int(y_min)},{int(x_max)},{int(y_max)}) "
                              f"score=0 raw='{cleaned}' (invalid)")
            continue
        if not (0.0003*img_area < area < 0.20*img_area):
            if verbose: print("[OCR BOX] area reject", cleaned); continue
        if not (3.0 < aspect < 10.0):
            if verbose: print("[OCR BOX] aspect reject", cleaned); continue
        if cy < 0.70 * H:
            if verbose: print("[OCR BOX] too high", cleaned); continue

        pos_score = max(0.0, 1.0 - abs(cy/H - 0.85))
        score = float(conf) + 0.6*pos_score + 0.05*min(len(cleaned), 10)

        if verbose:
            print(f"[OCR BOX] ({int(x_min)},{int(y_min)},{int(x_max)},{int(y_max)}) "
                  f"score={score:.3f} raw='{cleaned}'")

        if score > best_score:
            best_score = score
            best = (int(x_min), int(y_min), int(x_max), int(y_max), text, conf)

    return best, best_score

# ---------- Detection: contour-based fallback ----------

def detect_plate_by_contours(image_bgr, reader, verbose=True):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray_blur, 50, 200)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), 1)

    plt.figure(figsize=(8,5))
    plt.imshow(edges, cmap='gray'); plt.title("Canny + dilation"); plt.axis("off"); plt.show()

    cnts,_ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape; img_area = H*W
    best_box, best_score = None, 0.0

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area, aspect, cy = w*h, w/float(h), y + h/2.0
        if not (0.0003*img_area < area < 0.25*img_area): continue
        if not (2.0 < aspect < 10.0): continue
        if cy < 0.60 * H: continue

        roi = image_bgr[y:y+h, x:x+w]
        if roi.size == 0: continue
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_proc = preprocess_for_ocr(roi_gray)
        ocr = reader.readtext(roi_proc, detail=1,
                              allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        if not ocr: continue
        raw = "".join(d[1] for d in ocr)
        cleaned = clean_text(raw)
        if not text_is_valid_plate(cleaned):
            if verbose: print(f"[CNT BOX] ({x},{y},{x+w},{y+h}) score=0 raw='{cleaned}' (invalid)")
            continue
        avg_conf = float(np.mean([d[2] for d in ocr]))
        score = avg_conf + 0.05*min(len(cleaned),10)

        if verbose:
            print(f"[CNT BOX] ({x},{y},{x+w},{y+h}) score={score:.3f} raw='{cleaned}'")

        if score > best_score:
            best_score = score
            best_box = (x, y, x+w, y+h, raw, avg_conf)

    return best_box, best_score

# ---------- OCR fusion helpers ----------

def best_full_image_plate_text(results):
    best, best_conf = "", 0.0
    for (_, text, conf) in results:
        cleaned = clean_text(text)
        if not text_is_valid_plate(cleaned): continue
        if conf > best_conf:
            best_conf = float(conf); best = cleaned
    return best, best_conf

def fuse_text(roi_clean, full_clean):
    roi, full = roi_clean, full_clean
    if not roi and not full: return ""
    if not roi: return full
    if not full: return roi
    if roi in full and len(full) >= len(roi): return full
    if full in roi and len(roi) >= len(full): return roi

    max_k = min(len(roi), len(full))
    for k in range(max_k, 1, -1):
        if roi[-k:] == full[-k:]:
            return full[:-k] + roi[-k:]

    candidates = [c for c in (roi, full) if text_is_valid_plate(c)]
    return max(candidates, key=len) if candidates else full

# ================= MAIN =================

print("📷 Please upload an image containing a vehicle number plate:")
uploaded = files.upload()
image_path = next(iter(uploaded.keys()))
print(f"✅ Image uploaded: {image_path}")

image = cv2.imread(image_path)
if image is None:
    raise ValueError("Error reading image.")

H,W = image.shape[:2]
if max(H,W) < 900:
    s = 900.0 / max(H,W)
    image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    H,W = image.shape[:2]
    print(f"[INFO] Upscaled image by {s:.2f}x")

plt.figure(figsize=(8,5))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image"); plt.axis("off"); plt.show()

reader = easyocr.Reader(['en'])

print("[INFO] Running EasyOCR on full image...")
results = reader.readtext(image)
print(f"[INFO] OCR boxes: {len(results)}")

# detection by OCR boxes and contours
ocr_box, ocr_score = select_best_text_box(results, image.shape, verbose=True)
cnt_box, cnt_score = detect_plate_by_contours(image, reader, verbose=True)

if ocr_box is not None and (ocr_score >= cnt_score):
    final_box, origin = ocr_box, "OCR_BOX"
elif cnt_box is not None:
    final_box, origin = cnt_box, "CONTOUR"
else:
    final_box, origin = None, None
print(f"[INFO] Chosen detection origin: {origin}")

roi_clean, roi_conf, have_plate_roi = "", 0.0, False

if final_box is not None:
    x1,y1,x2,y2,raw_box,conf_box = final_box
    print(f"[INFO] Selected plate box: ({x1},{y1},{x2},{y2}), origin={origin}")
    x1e,y1e,x2e,y2e = expand_roi(image, x1, y1, x2, y2, expand_ratio=0.40)
    plate_bgr = image[y1e:y2e, x1e:x2e].copy()
    have_plate_roi = plate_bgr.size > 0

    if have_plate_roi:
        plt.figure(figsize=(6,3))
        plt.imshow(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Separated Number Plate"); plt.axis("off"); plt.show()

        cv2.imwrite("separated_plate.png", plate_bgr)

        target_width = 400
        hp,wp = plate_bgr.shape[:2]
        sz = target_width/float(wp)
        plate_zoom = cv2.resize(plate_bgr, None, fx=sz, fy=sz,
                                interpolation=cv2.INTER_CUBIC)

        plt.figure(figsize=(6,3))
        plt.imshow(cv2.cvtColor(plate_zoom, cv2.COLOR_BGR2RGB))
        plt.title("Zoomed Number Plate"); plt.axis("off"); plt.show()

        gray_plate = cv2.cvtColor(plate_zoom, cv2.COLOR_BGR2GRAY)
        plate_proc = preprocess_for_ocr(gray_plate)

        plt.figure(figsize=(6,3))
        plt.imshow(plate_proc, cmap='gray')
        plt.title("Preprocessed Plate ROI"); plt.axis("off"); plt.show()

        ocr_roi = reader.readtext(
            plate_proc, detail=1,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        if ocr_roi:
            roi_raw = "".join(d[1] for d in ocr_roi)
            roi_clean = clean_text(roi_raw)
            roi_conf = float(np.mean([d[2] for d in ocr_roi]))
            print("[INFO] ROI raw:", roi_raw)
            print("[INFO] ROI cleaned:", roi_clean)
        else:
            roi_clean = clean_text(raw_box); roi_conf = float(conf_box)
else:
    print("[WARN] No plate-like region found.")

# fusion with full-image OCR
full_best, full_conf = best_full_image_plate_text(results)
print(f"[INFO] Full-image best text: '{full_best}', conf={full_conf:.3f}")

fused_clean = fuse_text(roi_clean, full_best)
print("[INFO] Fused cleaned text:", fused_clean)

if fused_clean and text_is_valid_plate(fused_clean):
    final_clean = postprocess_plate_text(fused_clean)
    ocr_conf = max(roi_conf, full_conf)
else:
    final_clean = ""
    ocr_conf = max(roi_conf, full_conf)

print("\n==============================")
print("Final detected plate:", final_clean if final_clean else "Plate not recognized")
print("OCR confidence:", round(ocr_conf*100, 2), "%")
print("==============================")

# confidence bar
plt.figure(figsize=(6,1))
plt.barh(["OCR Confidence"], [ocr_conf], color='green')
plt.xlim(0,1); plt.xlabel("Confidence (0–1)")
plt.title("OCR Confidence Level"); plt.show()

# final annotated image
annotated = image.copy()
label = final_clean if final_clean else "Plate not recognized"

if final_box is not None and final_clean and have_plate_roi:
    cv2.rectangle(annotated, (x1e,y1e), (x2e,y2e), (0,255,0), 2)
    cv2.putText(annotated, label, (x1e, max(0,y1e-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
else:
    cv2.putText(annotated, label, (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 2)

plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.title(f"Final result: {label}")
plt.axis("off")
 plt.show()