#  reimplementation of the configurator.py file from the nanochat repo 

import os
import sys
from ast import literal_eval

def _rank0_print(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

def _apply_config_file(path: str):
    _rank0_print(f"[Config] Applying config file: {path}")
    with open(path, "r") as f:
        _rank0_print(f.read())
    
    # exec in the caller's global scope
    exec(open(path).read(), globals(), globals())


def _parse_override(arg: str):
    # expects --key=value
    if not arg.startswith("--") or "=" not in arg:
        raise ValueError(f"Invalid override argument: {arg}, expected --key=value")
    key, value = arg[2:].split("=", 1) # allow '=' in value after the first one
    return key, value

def _coerce_value(value: str):
    try:
        return literal_eval(value)
    except Exception:
        return value

def _apply_override(key: str, value: str):
    if key not in globals():
        raise ValueError(f"Invalid key: {key}, not found in globals()")
    
    attempt = _coerce_value(value)
    current = globals()[key]
    if current is not None and type(attempt) != type(current):
        raise TypeError(
            f"Type mismatch for key: {key}, got {type(attempt).__name__}, expected {type(current).__name__}"
        )
    _rank0_print(f"[Config] Applied override: {key} = {attempt!r}")
    globals()[key] = attempt

def _main(argv):
    for arg in argv[1:]:
        if "=" not in arg:
            # treat as a config file path
            if arg.startswith("--"):
                raise ValueError(f"Flag looks like a file: {arg}. Use --key=value or a plain path.")
            _apply_config_file(arg)
        else:
            key, value = _parse_override(arg)
            _apply_override(key, value)

# when exec'd from train.py, run immediately
if __name__ == "__main__": # for debugging
    _main(sys.argv)
else:
    _main(sys.argv)