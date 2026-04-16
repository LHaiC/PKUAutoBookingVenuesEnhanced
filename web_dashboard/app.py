from flask import Flask, render_template, request, jsonify
import sys
import os

# 确保可以导入父目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

import json
import datetime
from cron import set_crontab, reset_crontab

STATUS_FILE = 'status.json'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({"status": "unknown", "last_run": None})

@app.route('/logs')
def logs():
    log_file = 'config0.log'
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return jsonify({"lines": lines[-100:]})
    except:
        return jsonify({"lines": []})

@app.route('/run', methods=['POST'])
def run_booking():
    import subprocess
    result = subprocess.run(
        ['python', 'main.py'],
        capture_output=True,
        text=True,
        timeout=120
    )
    return jsonify({
        "returncode": result.returncode,
        "stdout": result.stdout[-500:],
        "stderr": result.stderr[-500:]
    })

@app.route('/schedule', methods=['POST'])
def schedule():
    data = request.json
    action = data.get('action')
    hours = data.get('hours', 24)
    try:
        if action == 'enable':
            set_crontab(hours)
            return jsonify({"success": True, "message": f"定时任务已设置，每{hours}小时执行"})
        else:
            reset_crontab()
            return jsonify({"success": True, "message": "定时任务已清除"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)