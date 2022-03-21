#!/bin/bash
GPUS=$1
CONFIG=$2
MODEL=$3
PY_ARGS=${@:4}

set -x

python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    tools/train.py -c ${CONFIG} --model ${MODEL} ${PY_ARGS}
