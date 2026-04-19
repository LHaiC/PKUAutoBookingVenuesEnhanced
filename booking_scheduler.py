from __future__ import annotations

import argparse
import configparser
import datetime as dt
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_FILE = os.path.join(ROOT_DIR, "tasks.json")
SCHEDULER_STATUS_FILE = os.path.join(ROOT_DIR, "scheduler_status.json")


def load_tasks(path=TASKS_FILE):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("tasks", [])
    return data


def save_tasks(tasks, path=TASKS_FILE):
    payload = {"tasks": tasks}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_task(task):
    normalized = {
        key: task[key]
        for key in (
            "id",
            "name",
            "config",
            "browser",
            "enabled",
            "venue",
            "venue_num",
            "start_time",
            "end_time",
            "lead_seconds",
            "interval_seconds",
            "max_attempts",
            "timeout_seconds",
        )
        if key in task
    }
    normalized.setdefault("id", uuid.uuid4().hex[:12])
    normalized.setdefault("name", normalized.get("config", "config.ini"))
    normalized.setdefault("config", "config.ini")
    normalized.setdefault("browser", "firefox")
    normalized.setdefault("enabled", True)
    normalized.setdefault("venue", "")
    normalized.setdefault("venue_num", "-1")
    normalized.setdefault("start_time", "")
    normalized.setdefault("end_time", "")
    normalized.setdefault("lead_seconds", 60)
    normalized.setdefault("interval_seconds", 5)
    normalized.setdefault("max_attempts", 120)
    normalized.setdefault("timeout_seconds", 180)
    return normalized


def task_config_path(task):
    config_path = normalize_task(task)["config"]
    if os.path.isabs(config_path):
        return config_path
    return os.path.join(ROOT_DIR, config_path)


def read_booking_config(config_path):
    conf = configparser.ConfigParser()
    conf.read(config_path, encoding="utf8")
    if not conf.has_section("time"):
        raise ValueError(f"missing [time] section in {config_path}")
    return {
        "venue": conf.get("type", "venue", fallback=""),
        "venue_num": conf.get("type", "venue_num", fallback="-1"),
        "start_time": conf.get("time", "start_time", fallback="").strip(),
        "end_time": conf.get("time", "end_time", fallback="").strip(),
    }


def validate_account_config(config_path):
    conf = configparser.ConfigParser()
    conf.read(config_path, encoding="utf8")
    if not conf.has_section("login"):
        raise ValueError(f"missing [login] section in {config_path}")
    if not conf.get("login", "user_name", fallback="").strip():
        raise ValueError(f"missing [login] user_name in {config_path}")
    if not conf.get("login", "password", fallback="").strip():
        raise ValueError(f"missing [login] password in {config_path}")


def target_date_from_token(token, today=None):
    today = today or dt.date.today()
    date_part = token.split("-", 1)[0].strip()
    if len(date_part) == 8:
        return dt.datetime.strptime(date_part, "%Y%m%d").date()
    weekday = int(date_part)
    if weekday < 1 or weekday > 7:
        raise ValueError(f"weekday must be 1-7: {token}")
    delta_days = (weekday - 1 - today.weekday()) % 7
    if delta_days == 0:
        delta_days = 7
    return today + dt.timedelta(days=delta_days)


def release_datetime_for_token(token, today=None):
    target_date = target_date_from_token(token, today)
    release_date = target_date - dt.timedelta(days=3)
    return dt.datetime.combine(release_date, dt.time(12, 0))


def start_tokens_for_task(task):
    task = normalize_task(task)
    start_time = task.get("start_time", "").strip()
    if not start_time:
        booking_config = read_booking_config(task_config_path(task))
        start_time = booking_config["start_time"]
    return [token.strip() for token in start_time.split("/") if token.strip()]


def earliest_release_datetime(task, today=None):
    releases = [release_datetime_for_token(token, today) for token in start_tokens_for_task(task)]
    if not releases:
        raise ValueError("task config has no start_time")
    return min(releases)


def task_due(task, now=None):
    task = normalize_task(task)
    if not task.get("enabled", True):
        return False
    now = now or dt.datetime.now()
    release_at = earliest_release_datetime(task, now.date())
    start_at = release_at - dt.timedelta(seconds=int(task.get("lead_seconds", 60)))
    return now >= start_at


def build_main_command(task, today=None):
    task = normalize_task(task)
    release_at = earliest_release_datetime(task, today)
    command = [
        sys.executable,
        "main.py",
        "--config",
        task["config"],
        "--browser",
        task["browser"],
        "--once",
        "--wait-until",
        str(release_at),
    ]
    if task.get("venue"):
        command.extend(["--venue", task["venue"]])
    if task.get("venue_num") not in (None, ""):
        command.extend(["--venue-num", str(task["venue_num"])])
    if task.get("start_time"):
        command.extend(["--start-time", task["start_time"]])
    if task.get("end_time"):
        command.extend(["--end-time", task["end_time"]])
    return command


def describe_task(task, today=None):
    described = normalize_task(task)
    try:
        if not described.get("venue") or not described.get("start_time") or not described.get("end_time"):
            booking_config = read_booking_config(task_config_path(described))
            described["venue"] = described.get("venue") or booking_config["venue"]
            described["venue_num"] = described.get("venue_num") or booking_config["venue_num"]
            described["start_time"] = described.get("start_time") or booking_config["start_time"]
            described["end_time"] = described.get("end_time") or booking_config["end_time"]
        described["booking_start_time"] = described["start_time"]
        described["booking_end_time"] = described["end_time"]
        release_at = earliest_release_datetime(described, today)
        run_after = release_at - dt.timedelta(seconds=int(described.get("lead_seconds", 60)))
        described["release_at"] = str(release_at)
        described["booking_action_at"] = str(release_at)
        described["run_after"] = str(run_after)
        described["error"] = None
    except Exception as exc:
        described["venue"] = ""
        described["venue_num"] = ""
        described["booking_start_time"] = ""
        described["booking_end_time"] = ""
        described["release_at"] = ""
        described["booking_action_at"] = ""
        described["run_after"] = ""
        described["error"] = str(exc)
    return described


def write_scheduler_status(payload):
    payload = dict(payload)
    payload["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(SCHEDULER_STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_task_until_success(task):
    task = normalize_task(task)
    attempts = int(task.get("max_attempts", 120))
    interval = int(task.get("interval_seconds", 5))
    timeout = int(task.get("timeout_seconds", 180))
    try:
        validate_account_config(task_config_path(task))
        command = build_main_command(task)
    except Exception as exc:
        write_scheduler_status({
            "status": "invalid_config",
            "task_id": task["id"],
            "task_name": task["name"],
            "error": str(exc),
        })
        return False

    for attempt in range(1, attempts + 1):
        started_at = dt.datetime.now().isoformat(timespec="seconds")
        result = subprocess.run(
            command,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        write_scheduler_status({
            "status": "success" if success else "retrying",
            "task_id": task["id"],
            "task_name": task["name"],
            "attempt": attempt,
            "started_at": started_at,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        })
        if success:
            return True
        if attempt < attempts:
            time.sleep(interval)
    return False


def run_due_tasks_once(tasks):
    due_tasks = [normalize_task(task) for task in tasks if task_due(task)]
    if not due_tasks:
        write_scheduler_status({"status": "waiting", "due_tasks": 0})
        return 0
    with ThreadPoolExecutor(max_workers=len(due_tasks)) as executor:
        results = list(executor.map(run_task_until_success, due_tasks))
    return 0 if all(results) else 1


def scheduler_loop(tasks_path=TASKS_FILE, poll_seconds=10):
    write_scheduler_status({"status": "started", "pid": os.getpid()})
    while True:
        try:
            tasks = load_tasks(tasks_path)
            run_due_tasks_once(tasks)
        except Exception as exc:
            write_scheduler_status({"status": "error", "error": str(exc)})
        time.sleep(poll_seconds)


def run_cli():
    parser = argparse.ArgumentParser(description="Run PKU booking tasks when reservation opens")
    parser.add_argument("--tasks", default=TASKS_FILE)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()

    if args.loop:
        scheduler_loop(args.tasks, args.poll_seconds)
        return 0
    return run_due_tasks_once(load_tasks(args.tasks))


if __name__ == "__main__":
    raise SystemExit(run_cli())
