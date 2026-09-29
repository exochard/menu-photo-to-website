#!/usr/bin/env bash
# Download the text models (Apache-2.0) into models/: the PP-OCRv3 DB detector from the
# OpenCV Model Zoo and PaddlePaddle's PP-OCRv5 Latin recogniser, both pinned by revision.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
get() { [ -s "models/$3" ] || curl -fsSL -o "models/$3" "https://huggingface.co/$1/resolve/$2"; }
get opencv/text_detection_ppocr main/text_detection_en_ppocrv3_2023may.onnx text_detection_en_ppocrv3_2023may.onnx
get PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx 89d3a50e2c27e2e7cceeab0e944c25c807d5db4f/inference.onnx \
  latin_PP-OCRv5_mobile_rec.onnx
sha256sum -c <<'SUMS'
03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587  models/text_detection_en_ppocrv3_2023may.onnx
7888113072263cb471b93f66dd5e2ad70548dc526fa1ace760d0d973dd121498  models/latin_PP-OCRv5_mobile_rec.onnx
SUMS
