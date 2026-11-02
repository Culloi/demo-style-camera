#!/bin/bash
# hayao (CartoonGAN) 320 转换
set -e
cd /tmp/style320
/tmp/rknn232venv/bin/python3 tools/convert_style.py \
    --onnx models/onnx/cgan_hayao.onnx --out models/hayao.rknn \
    --size 320 320 --norm anime --verify
echo HAYAO_DONE
