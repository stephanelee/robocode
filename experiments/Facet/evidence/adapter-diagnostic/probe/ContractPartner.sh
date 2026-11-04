#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec "$JAVA_HOME/bin/java" -cp .:api.jar ContractPartner
