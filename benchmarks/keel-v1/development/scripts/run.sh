#!/bin/bash
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk-amd64; export PATH=$JAVA_HOME/bin:$PATH
# run.sh BOTDIR NAME ENEMY ROUNDS PORT OUTNAME [SEED]
cd /tmp/claude-0/-home-user-robocode/58807e69-2be7-44fe-b02d-dd28f305589a/scratchpad/dev
KEEL_SEED=${7:-0} KEEL_METRICS=/tmp/claude-0/-home-user-robocode/58807e69-2be7-44fe-b02d-dd28f305589a/scratchpad/dev/$6.markers timeout 3000 python3 /tmp/claude-0/-home-user-robocode/58807e69-2be7-44fe-b02d-dd28f305589a/scratchpad/dev/devgame.py $1 $3 $4 $5 /tmp/claude-0/-home-user-robocode/58807e69-2be7-44fe-b02d-dd28f305589a/scratchpad/dev/$6 > /dev/null 2>&1
echo "$6: $(python3 /tmp/claude-0/-home-user-robocode/58807e69-2be7-44fe-b02d-dd28f305589a/scratchpad/dev/analyze.py /tmp/claude-0/-home-user-robocode/58807e69-2be7-44fe-b02d-dd28f305589a/scratchpad/dev/$6 $2 2>&1)"
