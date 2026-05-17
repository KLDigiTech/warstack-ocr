from flask import Flask, request, jsonify
from PIL import Image
import easyocr
import numpy as np
import os

app = Flask(__name__)

reader = easyocr.Reader(['en'], gpu=False)

DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def crop(img, x1, y1, x2, y2):
    return img.crop((x1, y1, x2, y2))


def save_crop(img, name):
    path = os.path.join(DEBUG_DIR, f"{name}.png")
    img.save(path)
    print("DEBUG SAVED:", path)


def ocr_text(img):
    arr = np.array(img)
    return reader.readtext(arr, detail=0, paragraph=False)


def clean_text(lines):

    if not lines:
        return ""

    text = " ".join(lines)

    replacements = {
        "|": "",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        "(": "",
        ")": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()


def extract_number(lines):

    for line in lines:

        digits = ''.join(c for c in line if c.isdigit())

        if digits:
            return int(digits)

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
                "error": "No image uploaded"
            })

        file = request.files["image"]

        image = Image.open(file.stream).convert("RGB")

        width, height = image.size

        print(f"IMAGE SIZE: {width}x{height}")

        # =====================================================
        # TEAM PLACEMENT
        # =====================================================

        placement_crop = crop(
            image,
            int(width * 0.39),
            int(height * 0.07),
            int(width * 0.59),
            int(height * 0.19)
        )

        save_crop(placement_crop, "placement")

        placement_lines = ocr_text(placement_crop)

        print("PLACEMENT OCR:", placement_lines)

        placement = 0

        for line in placement_lines:

            upper = line.upper()

            if "1E" in upper or "1ER" in upper:
                placement = 1

        # =====================================================
        # TEAM KILLS
        # =====================================================

        kills_crop = crop(
            image,
            int(width * 0.34),
            int(height * 0.30),
            int(width * 0.42),
            int(height * 0.45)
        )

        save_crop(kills_crop, "team_kills")

        kills_lines = ocr_text(kills_crop)

        print("TEAM KILLS OCR:", kills_lines)

        squad_kills = extract_number(kills_lines)

        # =====================================================
        # PLAYERS
        # =====================================================

        player_zones = [

            {
                "name": "player1",

                # HolyPriest34
                "pseudo": (0.085, 0.545, 0.205, 0.590),

                # 3
                "kills": (0.165, 0.842, 0.182, 0.885),

                # 3
                "deaths": (0.205, 0.842, 0.222, 0.885)
            },

            {
                "name": "player2",

                # Dieu Ksk
                "pseudo": (0.285, 0.545, 0.405, 0.590),

                # 10
                "kills": (0.365, 0.842, 0.388, 0.885),

                # 2
                "deaths": (0.405, 0.842, 0.422, 0.885)
            },

            {
                "name": "player3",

                # Iteryum_
                "pseudo": (0.495, 0.545, 0.615, 0.590),

                # 4
                "kills": (0.575, 0.842, 0.592, 0.885),

                # 1
                "deaths": (0.615, 0.842, 0.632, 0.885)
            },

            {
                "name": "player4",

                # DR-JEKIL19
                "pseudo": (0.705, 0.545, 0.835, 0.590),

                # 2
                "kills": (0.785, 0.842, 0.802, 0.885),

                # 5
                "deaths": (0.825, 0.842, 0.842, 0.885)
            }
        ]

        players = []

        for zone in player_zones:

            # =====================================
            # PSEUDO
            # =====================================

            px1 = int(width * zone["pseudo"][0])
            py1 = int(height * zone["pseudo"][1])
            px2 = int(width * zone["pseudo"][2])
            py2 = int(height * zone["pseudo"][3])

            pseudo_crop = crop(image, px1, py1, px2, py2)

            save_crop(pseudo_crop, f"{zone['name']}_pseudo")

            pseudo_lines = ocr_text(pseudo_crop)

            print(f"{zone['name']} PSEUDO OCR:", pseudo_lines)

            pseudo = clean_text(pseudo_lines)

            # =====================================
            # KILLS
            # =====================================

            kx1 = int(width * zone["kills"][0])
            ky1 = int(height * zone["kills"][1])
            kx2 = int(width * zone["kills"][2])
            ky2 = int(height * zone["kills"][3])

            player_kills_crop = crop(image, kx1, ky1, kx2, ky2)

            save_crop(player_kills_crop, f"{zone['name']}_kills")

            player_kills_lines = ocr_text(player_kills_crop)

            print(f"{zone['name']} KILLS OCR:", player_kills_lines)

            kills = extract_number(player_kills_lines)

            # =====================================
            # DEATHS
            # =====================================

            dx1 = int(width * zone["deaths"][0])
            dy1 = int(height * zone["deaths"][1])
            dx2 = int(width * zone["deaths"][2])
            dy2 = int(height * zone["deaths"][3])

            deaths_crop = crop(image, dx1, dy1, dx2, dy2)

            save_crop(deaths_crop, f"{zone['name']}_deaths")

            deaths_lines = ocr_text(deaths_crop)

            print(f"{zone['name']} DEATHS OCR:", deaths_lines)

            deaths = extract_number(deaths_lines)

            players.append({
                "pseudo": pseudo,
                "kills": kills,
                "deaths": deaths,
                "kd": 0,
                "score": 0
            })

        # =====================================================
        # FINAL RESPONSE
        # =====================================================

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
    app.run(host="0.0.0.0", port=7860, debug=True)