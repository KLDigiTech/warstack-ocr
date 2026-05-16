import os
import re
import base64
import cv2
import pytesseract
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import requests
from io import BytesIO

app = Flask(__name__)

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_stats(text):
    stats = {
        'kills'  : None,
        'deaths' : None,
        'score'  : None,
        'kd'     : None,
    }

    lines = text.lower().split('\n')
    lines = [l.strip() for l in lines if l.strip()]

    for line in lines:
        if 'kill' in line or 'élimin' in line:
            numbers = re.findall(r'\d+', line)
            if numbers and stats['kills'] is None:
                stats['kills'] = int(numbers[0])

        if 'death' in line or 'mort' in line:
            numbers = re.findall(r'\d+', line)
            if numbers and stats['deaths'] is None:
                stats['deaths'] = int(numbers[0])

        if 'score' in line:
            numbers = re.findall(r'\d+', line)
            if numbers and stats['score'] is None:
                stats['score'] = int(numbers[0])

    if stats['kills'] is not None and stats['deaths'] is not None:
        deaths = stats['deaths'] if stats['deaths'] > 0 else 1
        stats['kd'] = round(stats['kills'] / deaths, 2)

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

        # Chargement de l'image
        if image_base64:
            img = load_image_from_base64(image_base64)
        else:
            img = load_image_from_url(image_url)

        if img is None:
            return jsonify({ 'error': 'Image invalide ou non décodable' }), 400

        # Prétraitement + OCR
        processed = preprocess_image(img)
        text = pytesseract.image_to_string(processed, config='--psm 6')

        # Extraction des stats
        stats = extract_stats(text)
        stats['raw_text'] = text.strip()

        # Réponse aplatie (kills, deaths, kd, score, raw_text à la racine)
        return jsonify({ 'success': True, **stats })

    except Exception as e:
        return jsonify({ 'error': str(e) }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
