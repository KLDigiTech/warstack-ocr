from flask import Flask, request, jsonify
from PIL import Image
import easyocr
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)

reader = easyocr.Reader(['en'], gpu=False)

DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

@app.route("/")
def home():
    return "WARSTACK OCR ONLINE"

@app.route("/ocr", methods=["POST"])
def ocr():

    print("\n========================", flush=True)
    print("WARSTACK OCR START", flush=True)
    print("========================", flush=True)

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image"
        })

    file = request.files["image"]

    image = Image.open(file.stream).convert("RGB")
    image_np = np.array(image)

    h, w = image_np.shape[:2]

    print(f"IMAGE SIZE: {w}x{h}", flush=True)

    debug_name = f"{uuid.uuid4()}.png"
    debug_path = os.path.join(DEBUG_DIR, debug_name)

    cv2.imwrite(debug_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    print(f"DEBUG IMAGE SAVED: {debug_path}", flush=True)

    # =========================
    # TEAM KILLS
    # =========================

    kills_crop = image_np[
        int(h * 0.28):int(h * 0.42),
        int(w * 0.36):int(w * 0.48)
    ]

    kills_result = reader.readtext(kills_crop, detail=0)

    print("KILLS OCR:", kills_result, flush=True)

    squad_kills = 0

    for txt in kills_result:
        digits = ''.join(filter(str.isdigit, txt))

        if digits.isdigit():
            value = int(digits)

            if value < 200:
                squad_kills = value
                break

    # =========================
    # PLACEMENT
    # =========================

    placement_crop = image_np[
        int(h * 0.08):int(h * 0.24),
        int(w * 0.34):int(w * 0.54)
    ]

    placement_result = reader.readtext(placement_crop, detail=0)

    print("PLACEMENT OCR:", placement_result, flush=True)

    placement = 0

    for txt in placement_result:
        digits = ''.join(filter(str.isdigit, txt))

        if digits.isdigit():
            placement = int(digits)
            break

    # =========================
    # PLAYERS
    # =========================

    players = []

    player_boxes = [
        [0.08, 0.58, 0.24, 0.82],
        [0.25, 0.58, 0.41, 0.82],
        [0.42, 0.58, 0.58, 0.82],
        [0.59, 0.58, 0.75, 0.82]
    ]

    for i, box in enumerate(player_boxes):

        x1 = int(w * box[0])
        y1 = int(h * box[1])

        x2 = int(w * box[2])
        y2 = int(h * box[3])

        crop = image_np[y1:y2, x1:x2]

        result = reader.readtext(crop, detail=0)

        print(f"\nPLAYER {i+1}", flush=True)
        print(result, flush=True)

        pseudo = "UNKNOWN"
        kills = 0

        for txt in result:

            clean = txt.strip()

            # Ignore niveaux
            if clean.isdigit() and int(clean) > 50:
                continue

            # Pseudo probable
            if len(clean) >= 4 and any(c.isalpha() for c in clean):
                pseudo = clean.replace(" ", "")
                break

        for txt in result:

            digits = ''.join(filter(str.isdigit, txt))

            if digits.isdigit():

                val = int(digits)

                if val <= 50:
                    kills = val
                    break

        players.append({
            "pseudo": pseudo,
            "kills": kills,
            "deaths": 0,
            "kd": 0,
            "score": 0
        })

    response = {
        "success": True,
        "placement": placement,
        "squad_kills": squad_kills,
        "players": players
    }

    print("\nFINAL RESPONSE", flush=True)
    print(response, flush=True)

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)