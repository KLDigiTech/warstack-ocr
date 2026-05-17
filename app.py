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
    import re
    place_match = re.search(r'(\d+)[Ee][^\w]*(PLACE|place)', full_text)
    if place_match:
        stats['placement'] = int(place_match.group(1))

    # Scores individuels (188, 470, 188, 281)
    scores_match = re.findall(r'\b(1[0-9]{2}|[2-9][0-9]{2}|[1-9][0-9]{3})\b', full_text)
    
    # Pseudos
    pseudo_patterns = ['Inoxydable', 'Classifie', 'Sanguinaire', 'Sans limites', 
                       'HolyPriest', 'holypriest', 'lteryum', 'Iteryum',
                       'Dieu', 'IEKIL', 'DR ']

    # Stats ligne numérique finale — pattern: kills score deaths assists x4
    # Ligne: "7 715 3 B 5 10 505 8 1 7 9 935 8 1 3 9 100 a a 3"
    for line in lines_clean:
        nums = re.findall(r'\b(\d+)\b', line)
        if len(nums) >= 12:
            # 4 joueurs x (kills, score, deaths, assists)
            try:
                players = []
                for i in range(4):
                    base = i * 4
                    if base + 2 < len(nums):
                        players.append({
                            'kills'  : int(nums[base]),
                            'score'  : int(nums[base + 1]),
                            'deaths' : int(nums[base + 2]),
                        })
                if players:
                    stats['players'] = players
                    # HolyPriest34 = 3ème joueur (index 2)
                    me = players[2] if len(players) > 2 else players[0]
                    stats['kills']  = me['kills']
                    stats['deaths'] = me['deaths']
                    stats['score']  = me['score']
            except:
                pass

    # Kills escouade fallback
    elim_match = re.search(r'ELIMINATIONS CONFIRMEES[^\d]*(\d+)', full_text)
    if elim_match and stats['kills'] is None:
        stats['kills'] = int(elim_match.group(1))

    # KD
    if stats['kills'] is not None and stats['deaths'] is not None and stats['deaths'] > 0:
        stats['kd'] = round(stats['kills'] / stats['deaths'], 2)

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