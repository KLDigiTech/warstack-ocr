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
    import re

    stats = {
        'kills'     : None,
        'deaths'    : None,
        'score'     : None,
        'kd'        : None,
        'placement' : None,
        'players'   : [],
    }

    lines = text.split('\n')
    lines_clean = [l.strip() for l in lines if l.strip()]
    full_text = ' '.join(lines_clean)

    # Placement
    place_match = re.search(r'(\d+)[Ee]\s*PLACE', full_text, re.IGNORECASE)
    if place_match:
        stats['placement'] = int(place_match.group(1))

    # Ligne des pseudos + scores (ex: "Inoxydable 188 Classifie 470 Sanguinaire 188 Sans limites 281")
    pseudo_score_pattern = re.findall(r'([A-Za-z][A-Za-z0-9_\- ]{2,20}?)\s+(\d{3,4})', full_text)
    players_raw = [(p.strip(), int(s)) for p, s in pseudo_score_pattern if 100 <= int(s) <= 9999]

    # Ligne de stats numériques (kills score deaths assists x4)
    stats_line = None
    for line in lines_clean:
        nums = re.findall(r'\b(\d+)\b', line)
        if len(nums) >= 12:
            stats_line = nums
            break

    players = []
    if stats_line:
        for i in range(4):
            base = i * 4
            if base + 2 < len(stats_line):
                players.append({
                    'kills'  : int(stats_line[base]),
                    'score'  : int(stats_line[base + 1]),
                    'deaths' : int(stats_line[base + 2]),
                })

    # Associe pseudos + stats par position
    result_players = []
    for i, player in enumerate(players):
        pseudo = players_raw[i][0] if i < len(players_raw) else f'Joueur {i+1}'
        result_players.append({
            'pseudo' : pseudo,
            'kills'  : player['kills'],
            'score'  : player['score'],
            'deaths' : player['deaths'],
            'kd'     : round(player['kills'] / max(player['deaths'], 1), 2)
        })

    stats['players'] = result_players

    # Stats escouade globales
    elim_match = re.search(r'ELIMINATIONS CONFIRMEES\D*(\d+)', full_text)
    if elim_match:
        stats['kills'] = int(elim_match.group(1))

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