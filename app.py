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
# PREPROCESS IMAGE
# =====================================================

def preprocess_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresh

# =====================================================
# OCR TEXT EXTRACTION
# =====================================================

def extract_text(image):

    results = reader.readtext(image)

    texts = []

    for result in results:

        text = result[1]
        confidence = result[2]

        if confidence > 0.25:

            texts.append(text)

    return texts

# =====================================================
# PARSE SCOREBOARD
# =====================================================

def parse_scoreboard(texts):

    full_text = "\n".join(texts)

    placement = None
    squad_kills = None

    # =================================================
    # PLACEMENT
    # =================================================

    placement_match = re.search(
        r'(\d+)\s*(PLACE|PLACEMENT|PLACEMENT DE LESCOUADE)',
        full_text,
        re.IGNORECASE
    )

    if placement_match:

        placement = int(placement_match.group(1))

    # =================================================
    # KILLS ESCOUADE
    # =================================================

    big_numbers = re.findall(r'\b\d+\b', full_text)

    if big_numbers:

        numbers = [int(x) for x in big_numbers]

        filtered = [
            n for n in numbers
            if 0 <= n <= 200
        ]

        if filtered:

            filtered.sort()

            squad_kills = (
                filtered[-2]
                if len(filtered) > 1
                else filtered[0]
            )

    # =================================================
    # PLAYERS
    # =================================================

    players = []

    pseudo_regex = re.compile(
        r'^[A-Za-z0-9_\-\[\]]{3,24}$'
    )

    ignored_words = [
        'resultat',
        'vous',
        'placement',
        'progression',
        'escouade',
        'score',
        'kills'
    ]

    for i, text in enumerate(texts):

        clean = text.strip()

        # =========================
        # IGNORE UI WORDS
        # =========================

        if clean.lower() in ignored_words:
            continue

        # =========================
        # DETECT PSEUDO
        # =========================

        if pseudo_regex.match(clean):

            stats = []

            for j in range(i + 1, min(i + 6, len(texts))):

                nums = re.findall(r'\d+', texts[j])

                for n in nums:

                    stats.append(int(n))

            player = {

                "pseudo": clean,

                "kills": (
                    stats[0]
                    if len(stats) > 0
                    else 0
                ),

                "deaths": (
                    stats[1]
                    if len(stats) > 1
                    else 0
                ),

                "score": (
                    stats[2]
                    if len(stats) > 2
                    else 0
                )

            }

            # =====================
            # KD
            # =====================

            if player["deaths"] > 0:

                player["kd"] = round(
                    player["kills"] / player["deaths"],
                    2
                )

            else:

                player["kd"] = player["kills"]

            players.append(player)

    # =================================================
    # LIMIT PLAYERS
    # =================================================

    players = players[:4]

    # =================================================
    # RETURN JSON
    # =================================================

    return {

        "success": True,

        "placement": placement,

        "squad_kills": squad_kills,

        "players": players,

        "raw_text": texts

    }

# =====================================================
# OCR ROUTE
# =====================================================

@app.route('/ocr', methods=['POST'])
def ocr():

    try:

        # =============================================
        # CHECK FILE
        # =============================================

        if 'image' not in request.files:

            return jsonify({

                "success": False,
                "error": "Aucune image"

            }), 400

        file = request.files['image']

        # =============================================
        # READ IMAGE
        # =============================================

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
                "error": "Image invalide"

            }), 400

        # =============================================
        # IMAGE ZONES
        # =============================================

        h, w = img.shape[:2]

        placement_zone = img[
            0:int(h * 0.30),
            0:w
        ]

        players_zone = img[
            int(h * 0.45):h,
            0:w
        ]

        # =============================================
        # PREPROCESS
        # =============================================

        placement_processed = preprocess_image(
            placement_zone
        )

        players_processed = preprocess_image(
            players_zone
        )

        # =============================================
        # OCR
        # =============================================

        placement_texts = extract_text(
            placement_processed
        )

        players_texts = extract_text(
            players_processed
        )

        all_texts = (
            placement_texts +
            players_texts
        )

        # =============================================
        # PARSE
        # =============================================

        result = parse_scoreboard(
            all_texts
        )

        return jsonify(result)

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