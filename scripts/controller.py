#!/usr/bin/env python3
"""Control the complete EthLadder paper-trading stack.

Use this one command from Terminal instead of starting individual services:

    ./.venv/bin/python scripts/controller.py start
    ./.venv/bin/python scripts/controller.py restart
    ./.venv/bin/python scripts/controller.py status
    ./.venv/bin/python scripts/controller.py stop
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
DOMAIN = f"gui/{os.getuid()}"


@dataclass(frozen=True)
class Service:
    label: str
    plist_name: str

    @property
    def installed_plist(self) -> Path:
        return LAUNCH_AGENTS / self.plist_name

    @property
    def source_plist(self) -> Path:
        return REPO_ROOT / "scripts" / "launchd" / self.plist_name


# Start the optimizer before the web service. The web startup then requests a
# fresh cycle, waking an idle optimizer without waiting for its hourly timer.
SERVICES = (
    Service("com.ethladder.optimizer", "com.ethladder.optimizer.plist"),
    Service("com.ethladder.watcher", "com.ethladder.watcher.plist"),
    Service("com.ethladder.web", "com.ethladder.web.plist"),
)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def is_loaded(service: Service) -> bool:
    return run("launchctl", "print", f"{DOMAIN}/{service.label}", check=False).returncode == 0


def install_if_needed(service: Service) -> None:
    if service.installed_plist.exists():
        return
    if not service.source_plist.exists():
        raise FileNotFoundError(f"Missing launchd definition: {service.source_plist}")
    LAUNCH_AGENTS.mkdir(parents=True, exist_ok=True)
    service.installed_plist.write_bytes(service.source_plist.read_bytes())


def start_service(service: Service) -> None:
    install_if_needed(service)
    if not is_loaded(service):
        run("launchctl", "bootstrap", DOMAIN, str(service.installed_plist))
    run("launchctl", "kickstart", "-k", f"{DOMAIN}/{service.label}")
    print(f"started {service.label}")


def stop_service(service: Service) -> None:
    if is_loaded(service):
        run("launchctl", "bootout", f"{DOMAIN}/{service.label}")
        print(f"stopped {service.label}")


def show_status() -> int:
    failed = False
    for service in SERVICES:
        result = run("launchctl", "print", f"{DOMAIN}/{service.label}", check=False)
        if result.returncode:
            print(f"{service.label}: stopped")
            failed = True
            continue
        lines = [line.strip() for line in result.stdout.splitlines()]
        state = next((line for line in lines if line.startswith("state =")), "state = unknown")
        pid = next((line for line in lines if line.startswith("pid =")), "pid = unknown")
        print(f"{service.label}: {state}; {pid}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "restart", "stop", "status"))
    command = parser.parse_args().command

    if command == "status":
        return show_status()
    if command == "stop":
        for service in reversed(SERVICES):
            stop_service(service)
        return 0
    if command == "restart":
        for service in reversed(SERVICES):
            stop_service(service)
    for service in SERVICES:
        start_service(service)
    return show_status()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        print(f"Controller failed: {detail}", file=sys.stderr)
        raise SystemExit(1)
