#!/usr/bin/env bash
#
# A script to start the backend.

set -e

cd "$(dirname "$0")"

echo "=== Creating venv ==="
# preferably use python 3.9 here
python3 -m venv venv
source venv/bin/activate

echo "=== Installing dependecies ==="
pip --quiet install numpy pandas nltk yake torch scikit-learn gensim networkx torch-geometric flask

echo "=== Starting app ==="
python3 backend/src/app.py