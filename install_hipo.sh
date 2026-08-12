#!/bin/sh
set -eu

PYTHON="${PYTHON:-/opt/anaconda3/envs/psipm/bin/python}"
HIGHS_VERSION="$("$PYTHON" -c 'import highspy; print(highspy.Highs().version())')"

printf 'Installing highspy-extras %s for %s\n' "$HIGHS_VERSION" "$PYTHON"
"$PYTHON" -m pip install --upgrade "highspy-extras==$HIGHS_VERSION"
"$PYTHON" - <<'PY'
import highspy
import highspy_extras

print("highspy", highspy.Highs().version())
print("highspy-extras", highspy_extras.__version__)
print("library ABI", highspy_extras.get_library_version())
h = highspy.Highs()
h.setOptionValue("output_flag", False)
status = h.setOptionValue("solver", "hipo")
print("HiPO option status", status)
if str(status).endswith("kError"):
    raise SystemExit("HiPO is still unavailable")
if highspy.Highs().version() != highspy_extras.__version__:
    raise SystemExit("highspy and highspy-extras versions do not match")
print("HiPO installation verified")
PY
