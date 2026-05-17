from flask import Flask, request, jsonify
from PIL import Image
import easyocr
import numpy as np
import os
import uuid

app = Flask(__name__)

reader = easyocr.Reader(['en'], gpu=False)

DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)


# =========================================================
# OCR HELPERS
# =========================================================

def crop(img, x1, y1, x2, y2):
    return img.crop((x1, y1, x2, y2))


def save_crop(crop_img, name):
    path = os.path.join(DEBUG_DIR, f"{name}.png")
    crop_img.save(path)
    print(f"DEBUG SAVED: {path}")


def ocr_text(img):
    arr = np.array(img)
    result = reader.readtext(arr, detail=0, paragraph=False)
    return result


def clean_text(lines):
    return " ".join(lines).strip()


def extract_number(lines):
    for line in lines:
        digits = ''.join(c for c in line if c.isdigit())
        if digits:
            return int(digits)
    return 0


# =========================================================
# ROUTE
# =========================================================

@app.route("/ocr", methods=["POST"])
def ocr():

    try:

        print("\n======================")
        print("WARSTACK OCR START")
        print("======================")

        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image"
            })

        file = request.files['image']

        image = Image.open(file.stream).convert("RGB")

        width, height = image.size

        print(f"IMAGE SIZE: {width}x{height}")

        # =====================================================
        # TEAM PLACEMENT
        # =====================================================

        placement_crop = crop(
            image,
            int(width * 0.40),
            int(height * 0.11),
            int(width * 0.58),
            int(height * 0.24)
        )

        save_crop(placement_crop, "placement")

        placement_lines = ocr_text(placement_crop)

        print("PLACEMENT OCR:", placement_lines)

        placement = 0

        for line in placement_lines:
            if "1E" in line or "1ER" in line or "1EPLACE" in line:
                placement = 1
                break

        # =====================================================
        # TEAM KILLS
        # =====================================================

        team_kills_crop = crop(
            image,
            int(width * 0.34),
            int(height * 0.30),
            int(width * 0.42),
            int(height * 0.45)
        )

        save_crop(team_kills_crop, "team_kills")

        team_kills_lines = ocr_text(team_kills_crop)

        print("TEAM KILLS OCR:", team_kills_lines)

        squad_kills = extract_number(team_kills_lines)

        # =====================================================
        # PLAYERS
        # =====================================================

        player_zones = [
            {
                "name": "player1",
                "pseudo": (0.11, 0.61, 0.24, 0.69),
                "kills": (0.18, 0.82, 0.21, 0.90),
                "deaths": (0.22, 0.82, 0.25, 0.90)
            },
            {
                "name": "player2",
                "pseudo": (0.31, 0.61, 0.44, 0.69),
                "kills": (0.38, 0.82, 0.41, 0.90),
                "deaths": (0.42, 0.82, 0.45, 0.90)
            },
            {
                "name": "player3",
                "pseudo": (0.52, 0.61, 0.65, 0.69),
                "kills": (0.59, 0.82, 0.62, 0.90),
                "deaths": (0.63, 0.82, 0.66, 0.90)
            },
            {
                "name": "player4",
                "pseudo": (0.72, 0.61, 0.85, 0.69),
                "kills": (0.79, 0.82, 0.82, 0.90),
                "deaths": (0.83, 0.82, 0.86, 0.90)
            }
        ]

        players = []

        for zone in player_zones:

            # =========================================
            # PSEUDO
            # =========================================

            px1 = int(width * zone["pseudo"][0])
            py1 = int(height * zone["pseudo"][1])
            px2 = int(width * zone["pseudo"][2])
            py2 = int(height * zone["pseudo"][3])

            pseudo_crop = crop(image, px1, py1, px2, py2)

            save_crop(pseudo_crop, f"{zone['name']}_pseudo")

            pseudo_lines = ocr_text(pseudo_crop)

            print(f"{zone['name']} pseudo OCR:", pseudo_lines)

            pseudo = clean_text(pseudo_lines)

            # =========================================
            # KILLS
            # =========================================

            kx1 = int(width * zone["kills"][0])
            ky1 = int(height * zone["kills"][1])
            kx2 = int(width * zone["kills"][2])
            ky2 = int(height * zone["kills"][3])

            kills_crop = crop(image, kx1, ky1, kx2, ky2)

            save_crop(kills_crop, f"{zone['name']}_kills")

            kills_lines = ocr_text(kills_crop)

            print(f"{zone['name']} kills OCR:", kills_lines)

            kills = extract_number(kills_lines)

            # =========================================
            # DEATHS
            # =========================================

            dx1 = int(width * zone["deaths"][0])
            dy1 = int(height * zone["deaths"][1])
            dx2 = int(width * zone["deaths"][2])
            dy2 = int(height * zone["deaths"][3])

            deaths_crop = crop(image, dx1, dy1, dx2, dy2)

            save_crop(deaths_crop, f"{zone['name']}_deaths")

            deaths_lines = ocr_text(deaths_crop)

            print(f"{zone['name']} deaths OCR:", deaths_lines)

            deaths = extract_number(deaths_lines)

            players.append({
                "pseudo": pseudo,
                "kills": kills,
                "deaths": deaths,
                "kd": 0,
                "score": 0
            })

        # =====================================================
        # FINAL
        # =====================================================

        response = {
            "success": True,
            "placement": placement,
            "squad_kills": squad_kills,
            "players": players
        }

        print("\nFINAL RESPONSE:")
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