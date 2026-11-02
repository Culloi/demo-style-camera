#!/bin/bash
# 320x320 批量重转 7 个风格模型 (编译机后台执行, rknn-toolkit2 2.3.2 venv)
# anime 系 (hayao/sketch/ukiyoe/ink): mean/std=127.5, out [-1,1]
# zoo 系 (candy/mosaic/oil): imagenet norm, out 0~255
set -e
cd /tmp/style320
PY=/tmp/rknn232venv/bin/python3
C=tools/convert_style.py

conv() {  # conv <onnx> <rknn名> <norm>
    echo "===== $1 -> $2.rknn (norm=$3) ====="
    $PY $C --onnx "models/onnx/$1" --out "models/$2.rknn" --size 320 320 \
        --norm "$3" --verify
}

conv sketch.onnx          sketch   anime
conv candy_9.onnx         candy    imagenet
conv mosaic_9.onnx        mosaic   imagenet
conv rain_princess_9.onnx oil      imagenet
conv anime_hayao.onnx     hayao    anime
conv ukiyoe.onnx          ukiyoe   anime
conv ink.onnx             ink      anime
echo CONVERT_ALL_DONE
ls -l models/*.rknn
