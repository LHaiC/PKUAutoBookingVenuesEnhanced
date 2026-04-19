from __future__ import annotations

import configparser
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request


SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(os.environ.get("PKU_BOOKING_ROOT", os.getcwd())).resolve()
if not (ROOT_DIR / "main.py").exists():
    ROOT_DIR = SOURCE_ROOT
sys.path.insert(0, str(ROOT_DIR))

from booking_scheduler import (  # noqa: E402
    SCHEDULER_STATUS_FILE,
    TASKS_FILE,
    describe_task,
    load_tasks,
    normalize_task,
    save_tasks,
)


app = Flask(__name__)
STATUS_FILE = ROOT_DIR / "status.json"
SCHEDULER_PID_FILE = ROOT_DIR / ".scheduler.pid"
CONFIG_FIELDS = [
    ("login", "user_name"),
    ("login", "password"),
    ("wechat", "wechat_notice"),
    ("wechat", "SCKEY"),
    ("chaojiying", "username"),
    ("chaojiying", "password"),
    ("chaojiying", "soft_id"),
    ("glm_ocr", "enabled"),
    ("glm_ocr", "endpoint"),
    ("glm_ocr", "timeout"),
    ("glm_ocr", "allow_chaojiying_fallback"),
]


def project_path(path_value):
    path_value = path_value or "config.ini"
    candidate = (ROOT_DIR / path_value).resolve()
    if ROOT_DIR not in candidate.parents and candidate != ROOT_DIR:
        raise ValueError("path must stay inside project directory")
    return candidate


def ensure_config(path_value):
    path = project_path(path_value)
    if not path.exists():
        example = ROOT_DIR / "config.example.ini"
        if not example.exists():
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(example, path)
    return path


def read_config(path_value="config.ini"):
    path = ensure_config(path_value)
    conf = configparser.ConfigParser()
    conf.read(path, encoding="utf8")
    values = {}
    for section, option in CONFIG_FIELDS:
        values.setdefault(section, {})[option] = conf.get(section, option, fallback="")
    return {"path": str(path.relative_to(ROOT_DIR)), "values": values}


def write_config(path_value, values):
    path = ensure_config(path_value)
    conf = configparser.ConfigParser()
    conf.read(path, encoding="utf8")
    for legacy_section in ("type", "time"):
        conf.remove_section(legacy_section)
    for section, options in values.items():
        if section in ("type", "time"):
            continue
        if not conf.has_section(section):
            conf.add_section(section)
        for option, value in options.items():
            conf.set(section, option, str(value))
    with open(path, "w", encoding="utf8") as f:
        conf.write(f)
    return read_config(str(path.relative_to(ROOT_DIR)))


def read_json_file(path, fallback):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def config_summary(path):
    conf = configparser.ConfigParser()
    conf.read(path, encoding="utf8")
    return {
        "path": str(path.relative_to(ROOT_DIR)),
        "legacy_booking": conf.has_section("type") or conf.has_section("time"),
    }


def list_config_files():
    candidates = []
    for path in [ROOT_DIR / "config.ini", *ROOT_DIR.glob("config_*.ini"), *(ROOT_DIR / "configs").glob("*.ini")]:
        if path.exists() and path.name != "config.example.ini":
            candidates.append(path.resolve())
    unique = sorted(set(candidates), key=lambda item: str(item.relative_to(ROOT_DIR)))
    configs = []
    for path in unique:
        try:
            summary = config_summary(path)
            summary["error"] = None
        except Exception as exc:
            summary = {
                "path": str(path.relative_to(ROOT_DIR)),
                "legacy_booking": False,
                "error": str(exc),
            }
        configs.append(summary)
    return configs


def scheduler_pid():
    try:
        return int(SCHEDULER_PID_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def scheduler_running():
    pid = scheduler_pid()
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config", methods=["GET", "POST"])
def config_api():
    if request.method == "GET":
        return jsonify(read_config(request.args.get("path", "config.ini")))
    payload = request.get_json(force=True)
    return jsonify(write_config(payload.get("path", "config.ini"), payload.get("values", {})))


@app.route("/api/configs")
def configs_api():
    return jsonify({"configs": list_config_files()})


@app.route("/api/tasks", methods=["GET", "POST"])
def tasks_api():
    if request.method == "GET":
        return jsonify({"tasks": [describe_task(task) for task in load_tasks(TASKS_FILE)]})
    payload = request.get_json(force=True)
    tasks = [normalize_task(task) for task in payload.get("tasks", [])]
    save_tasks(tasks, TASKS_FILE)
    return jsonify({"tasks": [describe_task(task) for task in tasks]})


@app.route("/api/status")
def status_api():
    scheduler = read_json_file(SCHEDULER_STATUS_FILE, {"status": "stopped"})
    pid = scheduler_pid()
    running = scheduler_running()
    if pid and running:
        scheduler["pid"] = pid
    elif scheduler.get("status") in {"started", "waiting", "retrying"}:
        scheduler = dict(scheduler)
        scheduler["status"] = "stopped"
        scheduler.pop("pid", None)
    return jsonify({
        "booking": read_json_file(STATUS_FILE, {"status": "unknown", "last_run": None}),
        "scheduler": scheduler,
        "scheduler_running": running,
    })


@app.route("/api/logs")
def logs_api():
    config_path = request.args.get("config", "config.ini")
    log_path = project_path(config_path).with_suffix(".log")
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    return jsonify({"lines": lines[-120:]})


@app.route("/api/run", methods=["POST"])
def run_booking_api():
    payload = request.get_json(force=True)
    config_path = payload.get("config", "config.ini")
    browser = payload.get("browser", "firefox")
    command = [sys.executable, "main.py", "--config", config_path, "--browser", browser, "--once"]
    for payload_key, cli_key in (
        ("venue", "--venue"),
        ("venue_num", "--venue-num"),
        ("start_time", "--start-time"),
        ("end_time", "--end-time"),
    ):
        if payload.get(payload_key) not in (None, ""):
            command.extend([cli_key, str(payload[payload_key])])
    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=int(payload.get("timeout_seconds", 180)),
    )
    return jsonify({
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    })


@app.route("/api/scheduler/start", methods=["POST"])
def start_scheduler_api():
    if scheduler_running():
        return jsonify({"success": True, "message": "scheduler already running", "pid": scheduler_pid()})
    process = subprocess.Popen(
        [sys.executable, "booking_scheduler.py", "--tasks", "tasks.json", "--loop"],
        cwd=ROOT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    SCHEDULER_PID_FILE.write_text(str(process.pid), encoding="utf-8")
    return jsonify({"success": True, "pid": process.pid})


@app.route("/api/scheduler/stop", methods=["POST"])
def stop_scheduler_api():
    pid = scheduler_pid()
    if not pid:
        return jsonify({"success": True, "message": "scheduler not running"})
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    SCHEDULER_PID_FILE.unlink(missing_ok=True)
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
