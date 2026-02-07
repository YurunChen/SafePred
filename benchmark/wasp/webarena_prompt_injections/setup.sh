#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# unifies the setup scripts of the various agents and sets up the virtual environments in locations where our run scripts should expect them

# VisualWebarena
cd ../visualwebarena
echo "Installing required packages for visualwebarena..."
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
pip install -e .
deactivate

# set up prompt injector
cd ../webarena_prompt_injections
echo "Installing required packages for prompt injection tests..."
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
deactivate

# Claude-v3.5-sonnet
# Please replace with 37 if you want to evaluate with Claude-v3.7-sonnet with computer use
echo "Installing required packages for Claude computer use agent..."
cd ../claude-35-computer-use-demo
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r dev-requirements.txt
deactivate