#!/bin/bash
pip3 install -r requirements.txt
printf "import nltk\nnltk.download('punkt')" | python3
