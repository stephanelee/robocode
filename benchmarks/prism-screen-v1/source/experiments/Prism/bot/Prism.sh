#!/bin/sh
cd "$(dirname "$0")" || exit 1
exec java -Dfile.encoding=UTF-8 -XX:+UseSerialGC -Xms128m -Xmx512m -jar Prism.jar
