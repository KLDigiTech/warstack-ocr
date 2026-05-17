from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import easyocr
import numpy as np
import cv2
import uuid
import os
import re

app = Flask(__name__)
CORS(app)

reader = easyocr.Reader(['en'])

DEBUG_FOLDER = "debug"
os.makedirs(DEBUG_FOLDER, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def crop(image, x1, y1, x2, y2):
    return image[y1:y2, x1:x2]


def upscale(img, scale=3):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale))


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


def ocr_text(img):

    processed = preprocess(img)
    processed = upscale(processed, 3)

    results = reader.readtext(
        processed,
        detail=0,
        paragraph=False
    )

    cleaned = []

    for r in results:
        r = r.strip()

        if r:
            cleaned.append(r)

    return cleaned


def clean_pseudo(text):

    text = re.sub(r'[^A-Za-z0-9\-_]', '', text)

    blacklist = [
        "VOUS",
        "SANSLIMITES",
        "CLASSIFIE",
        "INOXYDABLE",
        "SANGUINAIRE",
        "RECUPEREES"
    ]

    upper = text.upper()

    for b in blacklist:
        if b in upper:
            return ""

    if len(text) < 3:
        return ""

    return text


def extract_number(texts):

    for t in texts:

        match = re.search(r'\d+', t)

        if match:
            return int(match.group())

    return 0


# =========================================================
# OCR ROUTE
# =========================================================

@app.route("/ocr", methods=["POST"])
def ocr():

    try:

        print("\n======================")
        print("WARSTACK OCR START")
        print("======================")

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image"
            })

        file = request.files["image"]

        image = Image.open(file.stream).convert("RGB")
        image = np.array(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        height, width = image.shape[:2]

        print(f"IMAGE SIZE: {width}x{height}")

        debug_id = str(uuid.uuid4())

        # =========================================================
        # TEAM PLACEMENT
        # =========================================================

        placement_crop = crop(
            image,
            int(width * 0.35),
            int(height * 0.10),
            int(width * 0.65),
            int(height * 0.22)
        )

        cv2.imwrite(
            f"{DEBUG_FOLDER}/{debug_id}_placement.png",
            placement_crop
        )

        placement_text = ocr_text(placement_crop)

        print("PLACEMENT OCR:", placement_text)

        placement = 0

        for t in placement_text:

            upper = t.upper()

            if "PLACE" in upper:

                match = re.search(r'\d+', upper)

                if match:
                    placement = int(match.group())
                    break

        # =========================================================
        # TEAM KILLS
        # =========================================================

        kills_crop = crop(
            image,
            int(width * 0.33),
            int(height * 0.27),
            int(width * 0.43),
            int(height * 0.42)
        )

        cv2.imwrite(
            f"{DEBUG_FOLDER}/{debug_id}_teamkills.png",
            kills_crop
        )

        kills_text = ocr_text(kills_crop)

        print("KILLS OCR:", kills_text)

        squad_kills = extract_number(kills_text)

        # =========================================================
        # PLAYERS
        # =========================================================

        player_zones = [

            {
                "name": "player1",

                "pseudo": (0.095, 0.605, 0.290, 0.645),

                "kills": (0.142, 0.832, 0.168, 0.872),

                "deaths": (0.178, 0.832, 0.204, 0.872),
            },

            {
                "name": "player2",

                "pseudo": (0.310, 0.605, 0.500, 0.645),

                "kills": (0.344, 0.832, 0.370, 0.872),

                "deaths": (0.380, 0.832, 0.406, 0.872),
            },

            {
                "name": "player3",

                "pseudo": (0.520, 0.605, 0.710, 0.645),

                "kills": (0.552, 0.832, 0.578, 0.872),

                "deaths": (0.588, 0.832, 0.614, 0.872),
            },

            {
                "name": "player4",

                "pseudo": (0.730, 0.605, 0.920, 0.645),

                "kills": (0.762, 0.832, 0.788, 0.872),

                "deaths": (0.798, 0.832, 0.824, 0.872),
            },
        ]

        players = []

        # =========================================================
        # LOOP PLAYERS
        # =========================================================

        for i, zone in enumerate(player_zones):

            print(f"\nPLAYER {i+1}")

            # =====================================================
            # PSEUDO
            # =====================================================

            px1 = int(width * zone["pseudo"][0])
            py1 = int(height * zone["pseudo"][1])
            px2 = int(width * zone["pseudo"][2])
            py2 = int(height * zone["pseudo"][3])

            pseudo_crop = crop(
                image,
                px1,
                py1,
                px2,
                py2
            )

            cv2.imwrite(
                f"{DEBUG_FOLDER}/{debug_id}_pseudo_{i}.png",
                pseudo_crop
            )

            pseudo_texts = ocr_text(pseudo_crop)

            print("PSEUDO:", pseudo_texts)

            pseudo = ""

            for t in pseudo_texts:

                cleaned = clean_pseudo(t)

                if cleaned:
                    pseudo = cleaned
                    break

            if pseudo == "":
                pseudo = f"JOUEUR{i+1}"

            # =====================================================
            # KILLS
            # =====================================================

            kx1 = int(width * zone["kills"][0])
            ky1 = int(height * zone["kills"][1])
            kx2 = int(width * zone["kills"][2])
            ky2 = int(height * zone["kills"][3])

            kills_crop = crop(
                image,
                kx1,
                ky1,
                kx2,
                ky2
            )

            cv2.imwrite(
                f"{DEBUG_FOLDER}/{debug_id}_kills_{i}.png",
                kills_crop
            )

            kills_texts = ocr_text(kills_crop)

            print("KILLS:", kills_texts)

            kills = extract_number(kills_texts)

            # =====================================================
            # DEATHS
            # =====================================================

            dx1 = int(width * zone["deaths"][0])
            dy1 = int(height * zone["deaths"][1])
            dx2 = int(width * zone["deaths"][2])
            dy2 = int(height * zone["deaths"][3])

            deaths_crop = crop(
                image,
                dx1,
                dy1,
                dx2,
                dy2
            )

            cv2.imwrite(
                f"{DEBUG_FOLDER}/{debug_id}_deaths_{i}.png",
                deaths_crop
            )

            deaths_texts = ocr_text(deaths_crop)

            print("DEATHS:", deaths_texts)

            deaths = extract_number(deaths_texts)

            # =====================================================
            # KD
            # =====================================================

            kd = 0

            if deaths > 0:
                kd = round(kills / deaths, 2)

            # =====================================================
            # PLAYER RESULT
            # =====================================================

            player = {
                "pseudo": pseudo,
                "kills": kills,
                "deaths": deaths,
                "kd": kd,
                "score": 0
            }

            print("FINAL PLAYER:", player)

            players.append(player)

        # =========================================================
        # FINAL RESPONSE
        # =========================================================

        response = {
            "success": True,
            "placement": placement,
            "squad_kills": squad_kills,
            "players": players
        }

        print("\nFINAL RESPONSE")
        print(response)

        return jsonify(response)

    except Exception as e:

        print("OCR ERROR:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        })


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )