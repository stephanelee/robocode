#!/bin/sh
set -eu
cd "$(dirname "$0")"
unset AISURVIVOR_DUEL_POLICY AISURVIVOR_ACTION AISURVIVOR_DATASET_PATH
export AISURVIVOR_MODE=train
export AISURVIVOR_SEED=20262121
export AISURVIVOR_MODEL_PATH=/home/runner/work/robocode/robocode/build/coherent-duel-v2/coherent-v2-Walls-previous/previous/AISurvivor/model.dat
export AISURVIVOR_METRICS_PATH=/home/runner/work/robocode/robocode/build/coherent-duel-v2/coherent-v2-Walls-previous/previous/AISurvivor/metrics.jsonl
exec /opt/hostedtoolcache/Java_Temurin-Hotspot_jdk/25.0.5-8/x64/bin/java -jar AISurvivor.jar
