from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import easyocr
import re

app = Flask(__name__)
CORS(app)

reader = easyocr.Reader(['en'], gpu=False)

# =====================================================
# UPSCALE
# =====================================================

def upscale(img, scale=4):

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

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = upscale(gray, 3)

    gray = cv2.GaussianBlur(
        gray,
        (3,3),
        0
    )

    return gray

# =====================================================
# PREPROCESS NUMBERS
# =====================================================

def preprocess_numbers(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = upscale(gray, 10)

    gray = cv2.GaussianBlur(
        gray,
        (3,3),
        0
    )

    thresh = cv2.threshold(
        gray,
        210,
        255,
        cv2.THRESH_BINARY
    )[1]

    return thresh

# =====================================================
# READ TEXT
# =====================================================

def read_text(img):

    results = reader.readtext(
        img,
        detail=0
    )

    text = ''.join(results)

    return text.strip()

# =====================================================
# READ NUMBER
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

        values = [int(n) for n in nums]

        valid = []

        for v in values:

            if v <= 100:
                valid.append(v)

        if valid:
            return max(valid)

    return 0

# =====================================================
# OCR ROUTE
# =====================================================

@app.route('/ocr', methods=['POST'])
def ocr():

    print("WARSTACK OCR ACTIVE")

    try:

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

        h, w = img.shape[:2]

        # =================================================
        # PLACEMENT
        # =================================================

        placement_zone = img[
            int(h * 0.12):int(h * 0.26),
            int(w * 0.33):int(w * 0.67)
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

        # =================================================
        # TEAM KILLS
        # =================================================

        kills_zone = img[
            int(h * 0.29):int(h * 0.38),
            int(w * 0.28):int(w * 0.39)
        ]

        squad_kills = read_number(
            preprocess_numbers(kills_zone)
        )

        # =================================================
        # PLAYERS
        # =================================================

        players = []

        player_columns = [

            [0.05, 0.22],
            [0.28, 0.45],
            [0.52, 0.69],
            [0.75, 0.92]

        ]

        for col in player_columns:

            x1 = int(w * col[0])
            x2 = int(w * col[1])

            # =============================================
            # NAME
            # =============================================

            name_crop = img[
                int(h * 0.60):int(h * 0.67),
                x1:x2
            ]

            pseudo = read_text(
                preprocess_text(name_crop)
            )

            pseudo = pseudo.replace(' ', '')

            pseudo = re.sub(
                r'[^A-Za-z0-9_\-]',
                '',
                pseudo
            )

            # =============================================
            # KILLS
            # =============================================

            kills_crop = img[
                int(h * 0.695):int(h * 0.745),
                x1 + 58:x1 + 95
            ]

            kills = read_number(
                preprocess_numbers(kills_crop)
            )

            if len(pseudo) < 3:
                continue

            players.append({

                "pseudo": pseudo,
                "kills": kills

            })

        return jsonify({

            "success": True,

            "placement": placement,

            "placement_text": placement_text,

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