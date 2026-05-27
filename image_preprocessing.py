import cv2
import numpy as np
from PIL import Image

def preprocess_image(pil_img, target_size=(224, 224)):
    # ---------------------------
    # 1. Ensure RGB
    # ---------------------------
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # ---------------------------
    # 2. Resize
    # ---------------------------
    pil_img = pil_img.resize(target_size, Image.BICUBIC)

    # ---------------------------
    # 3. Convert to NumPy (uint8)
    # ---------------------------
    img = np.array(pil_img).astype("uint8")

    # ---------------------------
    # 4. Noise reduction
    # ---------------------------
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # ---------------------------
    # 5. CLAHE on L channel (SAFE)
    # ---------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)        # uint8
    a = a.astype("uint8")    # FIX
    b = b.astype("uint8")    # FIX

    lab = cv2.merge((l, a, b))

    # ---------------------------
    # 6. Back to RGB
    # ---------------------------
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ---------------------------
    # 7. Normalize (MATCH TRAINING)
    # ---------------------------
    img = img.astype("float32") / 255.0

    # ---------------------------
    # 8. Add batch dimension
    # ---------------------------
    img = np.expand_dims(img, axis=0)

    return img
