#!/usr/bin/env bash

set -e
podman build -t mdok "$@" .
