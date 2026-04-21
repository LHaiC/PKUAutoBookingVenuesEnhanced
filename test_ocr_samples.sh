#!/bin/bash
set -e
cd /home/hcliu/projects/PKUAutoBookingVenues

python3 -c "
import base64, requests
from PIL import Image, ImageDraw
import io

session = requests.Session()
session.trust_env = False

samples = [
    ('captcha-20260420-143329-380473.png', ['今', '入', '心']),
    ('captcha-20260420-141157-545800.png', ['关', '历', '生']),
    ('captcha-20260420-142207-995092.png', ['打', '日', '合']),
    ('captcha-20260420-134057-253800.png', ['前', '思', '力']),
    ('captcha-20260420-171925-890793.png', ['叔', '领', '史']),
    ('captcha-20260420-172843-935131.png', ['长', '方', '科']),
]

for fname, targets in samples:
    img = Image.open(f'models/captcha_failures/{fname}')
    draw = ImageDraw.Draw(img)
    W, H = img.size

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()

    resp = session.post('http://localhost:8000/glmocr/parse',
        json={'images': [f'data:image/png;base64,{b64}'], 'targets': targets},
        timeout=120)
    resp.raise_for_status()
    r = resp.json()

    results = [x['text'] for x in r.get('results', [])]
    error = r.get('error', 'ok')
    detail = r.get('detail', '')
    status = 'PASS' if set(results) == set(targets) else 'FAIL'

    print(f'=== {fname} targets={targets} ===')
    print(f'  raw_outputs : {r.get(\"raw_outputs\", [])}')
    print(f'  raw_output  : {r.get(\"raw_output\", \"\")}')
    print(f'  results     : {results}')
    print(f'  error       : {error}')
    print(f'  detail      : {detail}')
    print(f'  method      : {r.get(\"method\")}')
    print(f'  image_size  : {r.get(\"image_size\")}')

    colors = ['red', 'lime', 'yellow', 'orange', 'cyan', 'magenta']
    for i, res in enumerate(r.get('results', [])):
        bbox = res.get('bbox', [])
        if bbox and len(bbox) == 4:
            color = colors[i % len(colors)]
            draw.rectangle(bbox, outline=color, width=2)
            draw.text((bbox[0], bbox[3]+2), f'{i}:{res[\"text\"]}', fill=color)
            print(f'  result[{i}] bbox={bbox} text={res[\"text\"]} confidence={res.get(\"confidence\")}')

    draw.text((5, 5), status, fill='green' if status=='PASS' else 'red')
    img.save(f'ocr_debug_{fname}')
    print(f'  -> {status}')
    print()
"
