"""
Utilities for generating training report cards. More messy code than usual, will fix.
"""


import os
import re
import shutil
import subprocess
import socket
import datetime
import platform
import psutil
import torch


def run_command(cmd):
    """ Run a shell command and return output, or None if it fails. """

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except:
        return None


def get_git_info():
    """ Get the current git commit, branch, and dirty status. """
    info = {}
    # whats rev-parse? git rev-parse functionality: show abbreviated commit hash of the current commit
    info['commit'] = run_command("git rev-parse --short HEAD") or "unknown"
    info['branch'] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"

    # check if repo is dirty (has uncommitted changes)
    status = run_command("git status --porcelain")
    info['dirty'] = bool(status) if status is not None else False

    # get commit message
    info['message'] = run_command("git log -1 --pretty=%B") or ""
    info['message'] = info['message'].split('\n')[0][:80]  # First line, truncated

    return info

def get_gpu_info():
    """ Get the GPU information. """
    if not torch.cuda.is_available():
         return {"available": False}

    num_devices = torch.cuda.device_count()
    info = {
        "available": True,
        "count": num_devices,
        "names": [],
        "memory_gb": []
    }

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info['names'].append(props.name)
        info['memory_gb'].append(props.total_memory / (1024 ** 3))

    # get cuda version
    info['cuda_version'] = torch.version.cuda or "unknown"

    return info


def get_system_info():
    """ get system information """ 
    info = {}
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.platform()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__
    
    # cpu and memory
    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    info['memory_gb'] = psutil.virtual_memory().total / (1024 ** 3)

    # user and env
    info['user'] = os.environ.get('USER', 'unknown')
    info['nanochat_base_dir'] = os.environ.get('NANOCHAT_BASE_DIR', 'out')
    info['working_dir'] = os.getcwd()

    return info

def estimate_cost(gpu_info, runtime_hours=None):
    """Estimate training cost based on GPU type and runtime."""
    # Rough pricing, from Lambda Cloud
    default_rate = 2.0
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    if not gpu_info.get("available"):
        return None

    # Try to identify GPU type from name
    hourly_rate = None
    gpu_name = gpu_info["names"][0] if gpu_info["names"] else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            hourly_rate = rate * gpu_info["count"]
            break
    
    if hourly_rate is None:
        hourly_rate = default_rate * gpu_info["count"]  # Default estimate

    return {
        "hourly_rate": hourly_rate,
        "gpu_type": gpu_name,
        "estimated_total": hourly_rate * runtime_hours if runtime_hours else None
    }


