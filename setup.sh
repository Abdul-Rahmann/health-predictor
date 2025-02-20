#!/bin/bash

PROJECT_ROOT=$(dirname "$(realpath "$0")")

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

echo "PYTHONPATH set to: $PYTHONPATH"
#echo "Run 'source setup.sh' before running the project."
echo "Project root set successfully!"