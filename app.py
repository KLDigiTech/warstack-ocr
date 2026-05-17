from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import easyocr
import re

app = Flask(__name__)
CORS(app)

# =====================================================
# EASYOCR
# =====================================================

reader = easyocr.Reader(['en'], gpu=False)

# =====================================================
# HELPERS
# =====================================================

def upscale(img, scale=3):

    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

# =====================================================
# PREPROCESS TEXT
# =====================================================

def preprocess_text(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = upscale(gray, 3)

    gray = cv2.GaussianBlur(gray, (3,3), 0)

    return gray

# =====================================================
# PREPROCESS NUMBERS
# =====================================================

def preprocess_numbers(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = upscale(gray, 4)

    thresh = cv2.threshold(
        gray,
        170,
        255,
        cv2.THRESH_BINARY
    )[1]

    return thresh

# =====================================================
# OCR TEXT
# =====================================================

def read_text(img):

    results = reader.readtext(
        img,
        detail=0
    )

    return " ".join(results).strip()

# =====================================================
# OCR NUMBER
# =====================================================

def read_number(img):

    results = reader.readtext(
        img,
        detail=0,
        allowlist='0123456789'
    )

    text = ''.join(results)

    nums = re.findall(r'\d+', text)

    if nums:
        return int(nums[0])

    return 0

# =====================================================
# OCR ROUTE
# =====================================================

@app.route('/ocr', methods=['POST'])
def ocr():

    try:

        # =============================================
        # CHECK IMAGE
        # =============================================

        if 'image' not in request.files:

            return jsonify({
                "success": False,
                "error": "No image"
            }), 400

        file = request.files['image']

        image_bytes = np.frombuffer(
            file.read(),
            np.uint8
        )

        img = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if img is None:

            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400

        # =============================================
        # IMAGE SIZE
        # =============================================

        h, w = img.shape[:2]

        # =============================================
        # PLACEMENT
        # =============================================

        placement_zone = img[
            int(h * 0.10):int(h * 0.28),
            int(w * 0.28):int(w * 0.72)
        ]

        placement_text = read_text(
            preprocess_text(placement_zone)
        )

        placement = None

        placement_match = re.search(
            r'(\d+)',
            placement_text
        )

        if placement_match:

            placement = int(
                placement_match.group(1)
            )

        # =============================================
        # TEAM KILLS
        # =============================================

        kills_zone = img[
            int(h * 0.30):int(h * 0.47),
            int(w * 0.28):int(w * 0.48)
        ]

        squad_kills = read_number(
            preprocess_numbers(kills_zone)
        )

        # =============================================
        # PLAYERS
        # =============================================

        players = []

        player_zones = [

            # PLAYER 1
            {
                "name": [
                    int(h * 0.60),
                    int(h * 0.68),
                    int(w * 0.03),
                    int(w * 0.20)
                ],
                "kills": [
                    int(h * 0.70),
                    int(h * 0.80),
                    int(w * 0.08),
                    int(w * 0.18)
                ]
            },

            # PLAYER 2
            {
                "name": [
                    int(h * 0.60),
                    int(h * 0.68),
                    int(w * 0.28),
                    int(w * 0.45)
                ],
                "kills": [
                    int(h * 0.70),
                    int(h * 0.80),
                    int(w * 0.32),
                    int(w * 0.42)
                ]
            },

            # PLAYER 3
            {
                "name": [
                    int(h * 0.60),
                    int(h * 0.68),
                    int(w * 0.52),
                    int(w * 0.69)
                ],
                "kills": [
                    int(h * 0.70),
                    int(h * 0.80),
                    int(w * 0.56),
                    int(w * 0.66)
                ]
            },

            # PLAYER 4
            {
                "name": [
                    int(h * 0.60),
                    int(h * 0.68),
                    int(w * 0.75),
                    int(w * 0.93)
                ],
                "kills": [
                    int(h * 0.70),
                    int(h * 0.80),
                    int(w * 0.79),
                    int(w * 0.89)
                ]
            }

        ]

        # =============================================
        # OCR PLAYERS
        # =============================================

        for zone in player_zones:

            # =========================================
            # NAME
            # =========================================

            y1, y2, x1, x2 = zone["name"]

            name_crop = img[y1:y2, x1:x2]

            pseudo = read_text(
                preprocess_text(name_crop)
            )

            pseudo = pseudo.replace(' ', '')

            # =========================================
            # KILLS
            # =========================================

            y1, y2, x1, x2 = zone["kills"]

            kills_crop = img[y1:y2, x1:x2]

            kills = read_number(
                preprocess_numbers(kills_crop)
            )

            # =========================================
            # CLEAN
            # =========================================

            if len(pseudo) < 2:
                continue

            players.append({

                "pseudo": pseudo,
                "kills": kills

            })

        # =============================================
        # RETURN
        # =============================================

        return jsonify({

            "success": True,

            "placement": placement,

            "squad_kills": squad_kills,

            "players": players

        })

    except Exception as e:

        print(e)

        return jsonify({

            "success": False,
            "error": str(e)

        }), 500

# =====================================================
# START
# =====================================================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=7860,
        debug=True
    )