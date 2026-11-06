@echo off
cd /d "%~dp0"
java -XX:+UseSerialGC -Xms128m -Xmx512m -jar KeelHelm.jar
