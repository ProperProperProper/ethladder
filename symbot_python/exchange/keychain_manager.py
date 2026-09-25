"""Keychain management — save, clear, and check API credentials.

Credentials are stored in macOS Keychain via the `security` CLI.
Never expose the actual credentials to the web layer — only report
whether they're set or not.
"""

from __future__ import annotations

import json
import subprocess
import logging

logger = logging.getLogger(__name__)

SERVICE = "unified-combo-grid"
ACCOUNT = "live"


def check_credentials_exist() -> bool:
    """Check if credentials are stored in Keychain without exposing them."""
    try:
        subprocess.check_output(
            ["security", "find-generic-password", "-s", SERVICE, "-a", ACCOUNT],
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def save_credentials(api_key: str, api_secret: str) -> dict:
    """Save API credentials to Keychain. Returns status."""
    if not api_key or not api_secret:
        return {"success": False, "error": "API key and secret are required"}

    try:
        creds = json.dumps({"api_key": api_key, "api_secret": api_secret})

        # Delete old entry if exists
        try:
            subprocess.run(
                ["security", "delete-generic-password", "-s", SERVICE, "-a", ACCOUNT],
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            )
        except Exception:
            pass  # Entry may not exist

        # Add new entry
        subprocess.run(
            ["security", "add-generic-password", "-s", SERVICE, "-a", ACCOUNT, "-w", creds],
            check=True,
            stderr=subprocess.PIPE,
            timeout=5.0,
        )
        logger.info("✓ API credentials saved to Keychain")
        return {"success": True, "message": "API credentials saved to Keychain"}
    except subprocess.CalledProcessError as e:
        error = e.stderr.decode() if e.stderr else str(e)
        logger.error("Failed to save credentials: %s", error)
        return {"success": False, "error": f"Keychain save failed: {error}"}
    except Exception as e:
        logger.error("Unexpected error saving credentials: %s", e)
        return {"success": False, "error": str(e)}


def clear_credentials() -> dict:
    """Remove API credentials from Keychain."""
    try:
        subprocess.run(
            ["security", "delete-generic-password", "-s", SERVICE, "-a", ACCOUNT],
            check=True,
            stderr=subprocess.PIPE,
            timeout=5.0,
        )
        logger.info("✓ API credentials cleared from Keychain")
        return {"success": True, "message": "API credentials cleared"}
    except subprocess.CalledProcessError as e:
        # Entry may not have existed
        if "could not find" in str(e).lower():
            logger.info("Credentials already cleared")
            return {"success": True, "message": "Credentials already cleared"}
        error = e.stderr.decode() if e.stderr else str(e)
        logger.error("Failed to clear credentials: %s", error)
        return {"success": False, "error": f"Keychain clear failed: {error}"}
    except Exception as e:
        logger.error("Unexpected error clearing credentials: %s", e)
        return {"success": False, "error": str(e)}
