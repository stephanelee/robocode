#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec java -XX:+UseSerialGC -Xms128m -Xmx512m -jar MeridianPeak.jar
