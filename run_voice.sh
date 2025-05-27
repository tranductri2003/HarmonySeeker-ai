#!/bin/bash
python -m venv venv_voice_separator
source venv_voice_separator/bin/activate
pip install -r requirements_voice.txt
python voice_api.py 