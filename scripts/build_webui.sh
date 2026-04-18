#!/usr/bin/env bash
set -euo pipefail

python -m PyInstaller \
  --onefile \
  --name pku-booking-webui \
  --add-data "web_dashboard/templates:web_dashboard/templates" \
  scripts/webui_launcher.py
