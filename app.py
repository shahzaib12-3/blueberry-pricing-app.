import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FAISS_INDEX = os.path.join(BASE_DIR, 'faiss_index.bin')
META_FILE = os.path.join(BASE_DIR, 'products_meta.json')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024

PASSWORD = "blueberry123"

print('Loading CLIP model...')
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

if not os.path.exists(FAISS_INDEX) or not os.path.exists(META_FILE):
    print('FAISS index or meta file not found. Run compute_embeddings.py first.')
    index = None
    meta = []
else:
    print('Loading FAISS index...')
    index = faiss.read_index(FAISS_INDEX)
    with open(META_FILE, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    print('Loaded index with', len(meta), 'items.')

def embed_image(image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        image_emb = model.get_image_features(**inputs)
    emb = image_emb.cpu().numpy()[0].astype('float32')
    emb = emb / np.linalg.norm(emb)
    return emb

@app.route('/', methods=['GET', 'POST'])
def index_route():
    if request.method == 'POST':
        password = request.form.get('password')
        if password != PASSWORD:
            return render_template('index.html', error="Incorrect password")
        if 'photo' not in request.files:
            return render_template('index.html', error="No photo uploaded")
        file = request.files['photo']
        try:
            image = Image.open(file.stream).convert('RGB')
        except Exception as e:
            return render_template('index.html', error="Could not open image: " + str(e))
        query_emb = embed_image(image)
        k = 5
        D, I = index.search(np.expand_dims(query_emb, axis=0), k)
        D = D[0].tolist()
        I = I[0].tolist()
        results = []
        for score, idx in zip(D, I):
            if idx < 0 or idx >= len(meta):
                continue
            item = meta[idx].copy()
            item['score'] = float(score)
            results.append(item)
        return render_template('index.html', match=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)