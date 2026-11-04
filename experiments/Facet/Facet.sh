#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec "${JAVA_HOME:+$JAVA_HOME/bin/}java" -Xms32m -Xmx256m -jar Facet.jar
