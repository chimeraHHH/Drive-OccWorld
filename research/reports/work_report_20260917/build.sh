#!/bin/sh
set -eu
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode -halt-on-error main.tex > build-pass1.log
xelatex -interaction=nonstopmode -halt-on-error main.tex > build-pass2.log
echo "Built main.pdf"
