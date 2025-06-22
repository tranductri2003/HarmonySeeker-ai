import os
import platform
import socket
import psutil
from datetime import datetime


def get_system_info():
    """
    Get system information for health reporting

    Returns:
        dict: System information including CPU, memory, disk usage
    """
    try:
        # Get CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count(logical=True)

        # Get memory info
        memory = psutil.virtual_memory()

        # Get disk info
        disk = psutil.disk_usage("/")

        # Get network info
        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "Unknown"

        # Get platform info
        platform_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

        # Get Python info
        python_info = {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "hostname": hostname,
            "ip_address": ip_address,
            "uptime": int(datetime.now().timestamp() - psutil.boot_time()),
            "cpu": {"percent": cpu_percent, "count": cpu_count},
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            },
            "platform": platform_info,
            "python": python_info,
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now().isoformat()}
