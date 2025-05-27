#!/bin/bash
python -m venv venv_chord_recognizer
source venv_chord_recognizer/bin/activate
pip install -r requirements_chord.txt
python chord_api.py 