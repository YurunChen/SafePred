#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
set -e

./start_all.sh
./novnc_startup.sh

python http_server.py > /tmp/server_logs.txt 2>&1 &

python -m computer_use_demo.cli "$@" > /dev/stdout 
