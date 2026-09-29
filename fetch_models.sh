#!/usr/bin/env bash
# Download the OpenCV Model Zoo text models (Apache-2.0) into models/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
base=https://huggingface.co/opencv
get() { [ -s "models/$2" ] || curl -fsSL -o "models/$2" "$base/$1/resolve/main/$2"; }
get text_detection_ppocr text_detection_en_ppocrv3_2023may.onnx
get text_recognition_crnn text_recognition_CRNN_CH_2023feb_fp16.onnx
sha256sum models/*.onnx
