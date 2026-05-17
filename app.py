from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import easyocr
import re

app = Flask(__name__)
CORS(app)

# =====================================================
# EASY OCR
# =====================================================

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

# =====================================================
# READ TEXT
# =====================================================

def read_text(img):

    results = reader.readtext(
        img,
        detail=0,
        paragraph=False,
        batch_size=1
    )

    return ' '.join(results)

# =====================================================
# READ NUMBER
# =====================================================

def read_number(img):

    results = reader.readtext(
        img,
        detail=0,
        paragraph=False,
        batch_size=1,
        allowlist='0123456789'
    )

    text = ''.join(results)

    nums = re.findall(r'\d+', text)

    if nums:

        try:
            return int(nums[0])
        except:
            return 0

    return 0

# =====================================================
# OCR ROUTE
# =====================================================

@app.route('/ocr', methods=['POST'])
def ocr():

    try:

        print("WARSTACK OCR START")

        # =================================================
        # IMAGE
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

        print(f"IMAGE SIZE {w}x{h}")

        # =================================================
        # PLACEMENT
        # =================================================

        placement_crop = img[
            int(h * 0.08):int(h * 0.20),
            int(w * 0.38):int(w * 0.62)
        ]

        placement_small = cv2.resize(
            placement_crop,
            (0, 0),
            fx=0.5,
            fy=0.5
        )

        placement_text = read_text(
            placement_small
        )

        print("PLACEMENT TEXT:", placement_text)

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

        kills_crop = img[
            int(h * 0.23):int(h * 0.37),
            int(w * 0.23):int(w * 0.40)
        ]

        kills_small = cv2.resize(
            kills_crop,
            (0, 0),
            fx=0.5,
            fy=0.5
        )

        squad_kills = read_number(
            kills_small
        )

        print("SQUAD KILLS:", squad_kills)

        # =================================================
        # PLAYERS
        # =================================================

        players = []

        player_zones = [

            (
                int(w * 0.03),
                int(h * 0.56),
                int(w * 0.23),
                int(h * 0.78)
            ),

            (
                int(w * 0.27),
                int(h * 0.56),
                int(w * 0.47),
                int(h * 0.78)
            ),

            (
                int(w * 0.51),
                int(h * 0.56),
                int(w * 0.71),
                int(h * 0.78)
            ),

            (
                int(w * 0.75),
                int(h * 0.56),
                int(w * 0.95),
                int(h * 0.78)
            )

        ]

        for index, zone in enumerate(player_zones):

            print(f"PLAYER {index + 1}")

            x1, y1, x2, y2 = zone

            crop = img[y1:y2, x1:x2]

            small = cv2.resize(
                crop,
                (0, 0),
                fx=0.5,
                fy=0.5
            )

            results = reader.readtext(
                small,
                detail=0,
                paragraph=False,
                batch_size=1
            )

            print(results)

            pseudo = "UNKNOWN"

            kills = 0

            # =============================================
            # PSEUDO
            # =============================================

            for text in results:

                clean = re.sub(
                    r'[^A-Za-z0-9_\-]',
                    '',
                    text
                )

                if len(clean) >= 3:

                    pseudo = clean
                    break

            # =============================================
            # KILLS
            # =============================================

            for text in results:

                nums = re.findall(r'\d+', text)

                if nums:

                    value = int(nums[0])

                    if value <= 50:

                        kills = value
                        break

            players.append({

                "pseudo": pseudo,
                "kills": kills,
                "deaths": 0,
                "kd": 0,
                "score": 0

            })

        # =================================================
        # RESPONSE
        # =================================================

        response = {

            "success": True,

            "placement": placement,

            "squad_kills": squad_kills,

            "players": players

        }

        print(response)

        return jsonify(response)

    except Exception as e:

        print("OCR ERROR:", str(e))

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