import os
import re
import base64
import cv2
import pytesseract
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_stats(text):
    stats = {
        'placement'  : None,
        'squad_kills': None,
        'players'    : [],
    }

    lines = text.split('\n')
    lines_clean = [l.strip() for l in lines if l.strip()]
    full_text = ' '.join(lines_clean)

    # Placement
    place_match = re.search(r'(\d+)[Ee]\s*PLACE', full_text, re.IGNORECASE)
    if place_match:
        stats['placement'] = int(place_match.group(1))

    # Kills escouade
    elim_match = re.search(r'ELIMINATIONS CONFIRMEES\D*(\d+)', full_text)
    if elim_match:
        stats['squad_kills'] = int(elim_match.group(1))

    # Ligne stats numériques — kills score deaths assists x4
    # Ex: "7 715 3 5 10 505 8 1 9 935 8 1 9 100 2 3"
    stats_nums = None
    for line in lines_clean:
        nums = re.findall(r'\b(\d+)\b', line)
        if len(nums) >= 12:
            stats_nums = [int(n) for n in nums]
            break

    # Pseudos — ligne avec noms joueurs
    # Ex: "Inoxydable 188 Classifie 470 Sanguinaire 188 Sans limites 281"
    pseudo_scores = []
    for line in lines_clean:
        # Cherche pattern: mot(s) suivi d'un nombre 3-4 chiffres, répété
        matches = re.findall(r'([A-Za-z][A-Za-z0-9_ ]{1,20}?)\s+(\d{3,4})\b', line)
        if len(matches) >= 3:
            pseudo_scores = [(m[0].strip(), int(m[1])) for m in matches[:4]]
            break

    # Construit les 4 joueurs
    players = []
    if stats_nums:
        for i in range(4):
            base = i * 4
            if base + 2 < len(stats_nums):
                pseudo = pseudo_scores[i][0] if i < len(pseudo_scores) else f'Joueur {i+1}'
                score  = pseudo_scores[i][1] if i < len(pseudo_scores) else stats_nums[base + 1]
                kills  = stats_nums[base]
                deaths = stats_nums[base + 2]
                players.append({
                    'pseudo' : pseudo,
                    'kills'  : kills,
                    'score'  : score,
                    'deaths' : deaths,
                    'kd'     : round(kills / max(deaths, 1), 2)
                })

    stats['players'] = players
    return stats

def load_image_from_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    img_array = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def load_image_from_base64(b64_string):
    img_data = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({ 'status': 'ok' })

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        data = request.get_json()
        if not data:
            return jsonify({ 'error': 'Body JSON manquant' }), 400

        image_url    = data.get('image_url')
        image_base64 = data.get('image_base64')

        if not image_url and not image_base64:
            return jsonify({ 'error': 'image_url ou image_base64 requis' }), 400

        if image_base64:
            img = load_image_from_base64(image_base64)
        else:
            img = load_image_from_url(image_url)

        if img is None:
            return jsonify({ 'error': 'Image invalide ou non décodable' }), 400

        processed = preprocess_image(img)
        text = pytesseract.image_to_string(processed, config='--psm 6')

        stats = extract_stats(text)
        stats['raw_text'] = text.strip()

        return jsonify({ 'success': True, **stats })

    except Exception as e:
        return jsonify({ 'error': str(e) }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)