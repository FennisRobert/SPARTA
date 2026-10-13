#!/bin/sh
# Run one assembled system through SPARTA and print timing and accuracy.
#
#     ./run_sparta.sh test_matrices/em_m3_asym2.sparta
#     ./run_sparta.sh test_matrices/em_m1_sym1.sparta --ldlt --verbose
#     ./run_sparta.sh test_matrices/em_m3_asym2.sparta --lu --refine 20 --minimize
#
# Uses the virtualenv that validate.sh builds under .local/venv when it exists,
# so the extension being measured is the one that was last installed there.
set -e
cd "$(dirname "$0")"

if [ -x .local/venv/bin/python ]; then
    PYTHON=.local/venv/bin/python
else
    PYTHON="${PYTHON:-python3}"
fi

exec "$PYTHON" tools/run_sparta.py "$@"
