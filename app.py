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
        'kills'  : None,
        'deaths' : None,
        'score'  : None,
        'kd'     : None,
    }

    lines = text.split('\n')
    lines = [l.strip() for l in lines if l.strip()]

    for i, line in enumerate(lines):
        line_lower = line.lower()

        # Kills — BF FR
        if any(k in line_lower for k in ['elimination', 'élimination', 'confirmee', 'confirmées']):
            numbers = re.findall(r'\b(\d{1,3})\b', line)
            if numbers and stats['kills'] is None:
                stats['kills'] = int(numbers[0])

        # Deaths
        if any(k in line_lower for k in ['death', 'mort', 'decede', 'décédé']):
            numbers = re.findall(r'\b(\d{1,3})\b', line)
            if numbers and stats['deaths'] is None:
                stats['deaths'] = int(numbers[0])

        # Score
        if 'score' in line_lower:
            numbers = re.findall(r'\b(\d+)\b', line)
            if numbers and stats['score'] is None:
                stats['score'] = int(numbers[0])

    # Fallback kills — cherche grands nombres isolés dans les premières lignes
    if stats['kills'] is None:
        for line in lines[:15]:
            numbers = re.findall(r'\b(\d{1,3})\b', line)
            if numbers:
                for n in numbers:
                    if 1 <= int(n) <= 100 and stats['kills'] is None:
                        stats['kills'] = int(n)
                        break

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