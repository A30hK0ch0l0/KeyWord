#!/usr/bin/env bash

## Change root path
cd "$(dirname "$0")/../" || return

## Config environments
VIRTUAL_ENV="$(pwd)/venv"
export VIRTUAL_ENV
PATH="$VIRTUAL_ENV/bin:$PATH"
export PATH
unset PYTHONHOME

## make .env file
cp -n .env.example .env

## Create virtual environment if not exists
if [ ! -d venv ]; then
    python3 -m venv venv  --copies
    python  -m pip install -U pip wheel setuptools
    python  -m pip install -r requirements.txt
fi

# run program
exec python -m pytest -s
