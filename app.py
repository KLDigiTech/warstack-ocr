from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import easyocr
import re

app = Flask(__name__)
CORS(app)

# =====================================================
# OCR MODEL
# =====================================================

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
        (3, 3),
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

    gray = upscale(gray, 8)

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
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

        # =================================================
        # CHECK IMAGE
        # =================================================

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
            int(h * 0.10):int(h * 0.28),
            int(w * 0.30):int(w * 0.70)
        ]

        placement_processed = preprocess_text(
            placement_zone
        )

        placement_text = read_text(
            placement_processed
        )

        placement = 0

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
            int(h * 0.28):int(h * 0.40),
            int(w * 0.28):int(w * 0.42)
        ]

        kills_processed = preprocess_numbers(
            kills_zone
        )

        squad_kills = read_number(
            kills_processed
        )

        # =================================================
        # PLAYERS
        # =================================================

        players = []

        player_zones = [

            # PLAYER 1
            (
                int(w * 0.03),
                int(h * 0.58),
                int(w * 0.23),
                int(h * 0.78)
            ),

            # PLAYER 2
            (
                int(w * 0.27),
                int(h * 0.58),
                int(w * 0.47),
                int(h * 0.78)
            ),

            # PLAYER 3
            (
                int(w * 0.51),
                int(h * 0.58),
                int(w * 0.71),
                int(h * 0.78)
            ),

            # PLAYER 4
            (
                int(w * 0.75),
                int(h * 0.58),
                int(w * 0.95),
                int(h * 0.78)
            )

        ]

        for zone in player_zones:

            x1, y1, x2, y2 = zone

            crop = img[y1:y2, x1:x2]

            processed = preprocess_text(crop)

            results = reader.readtext(
                processed,
                detail=0
            )

            pseudo = "UNKNOWN"

            kills = 0

            # =============================================
            # PSEUDO
            # =============================================

            if len(results) > 0:

                pseudo = results[0]

                pseudo = re.sub(
                    r'[^A-Za-z0-9_\-]',
                    '',
                    pseudo
                )

            # =============================================
            # KILLS
            # =============================================

            for text in results:

                nums = re.findall(
                    r'\d+',
                    text
                )

                if nums:

                    value = int(nums[0])

                    if value <= 50:

                        kills = value

                        break

            # =============================================
            # SAVE PLAYER
            # =============================================

            if len(pseudo) >= 3:

                players.append({

                    "pseudo": pseudo,

                    "kills": kills,

                    "deaths": 0,

                    "kd": 0,

                    "score": 0

                })

        # =================================================
        # RETURN
        # =================================================

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