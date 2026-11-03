#!/bin/sh
set -eu
cd "$(dirname "$0")"
unset AISURVIVOR_DUEL_POLICY AISURVIVOR_ACTION AISURVIVOR_DATASET_PATH
export AISURVIVOR_MODE=train
export AISURVIVOR_SEED=20262411
export AISURVIVOR_MODEL_PATH=/home/runner/work/robocode/robocode/build/final-default-redo/batch-2/dc946r2-b2-g01/candidate/AISurvivor/model.dat
export AISURVIVOR_METRICS_PATH=/home/runner/work/robocode/robocode/build/final-default-redo/batch-2/dc946r2-b2-g01/candidate/AISurvivor/metrics.jsonl
exec /opt/hostedtoolcache/Java_Temurin-Hotspot_jdk/25.0.5-8/x64/bin/java -jar AISurvivor.jar
