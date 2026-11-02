#!/bin/bash
# 剩余模型转换: ukiyoe / ink (hayao 需换 CartoonGAN 源, 另行处理)
set -e
cd /tmp/style320
PY=/tmp/rknn232venv/bin/python3
C=tools/convert_style.py

conv() {
    echo "===== $1 -> $2.rknn (norm=$3) ====="
    $PY $C --onnx "models/onnx/$1" --out "models/$2.rknn" --size 320 320 \
        --norm "$3" --verify
}

conv ukiyoe.onnx ukiyoe anime
conv ink.onnx     ink      anime
echo CONVERT_REMAIN_DONE
