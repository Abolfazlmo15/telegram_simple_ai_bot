#!/usr/bin/env python3
"""
Automatic renewal of PythonAnywhere web app (free accounts).
Logs in and clicks the "Run until 3 months from today" button.
Uses requests with session handling.
"""
import os
import sys
import logging
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Read credentials from environment variables (set in GitHub Secrets)
USERNAME = os.environ.get("PYTHONANYWHERE_USERNAME")
PASSWORD = os.environ.get("PYTHONANYWHERE_PASSWORD")
# Optional: specify the web app name (default is the first one)
WEBAPP_NAME = os.environ.get("PYTHONANYWHERE_WEBAPP_NAME", "www.pythonanywhere.com")


def get_csrf_token(session: requests.Session, url: str) -> Optional[str]:
    """Extract CSRF token from a page."""
    response = session.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    # Look for a hidden input named 'csrfmiddlewaretoken'
    token = soup.find("input", {"name": "csrfmiddlewaretoken"})
    if token:
        return token.get("value")
    # Alternative: look for meta tag or data attribute
    return None


def login(session: requests.Session) -> bool:
    """Log in to PythonAnywhere."""
    login_url = "https://www.pythonanywhere.com/login/"
    logger.info("Getting login page...")
    csrf_token = get_csrf_token(session, login_url)
    if not csrf_token:
        logger.error("Failed to get CSRF token from login page")
        return False

    logger.info("Logging in...")
    payload = {
        "csrfmiddlewaretoken": csrf_token,
        "username": USERNAME,
        "password": PASSWORD,
        "login_view-current_step": "auth",
        "next": "/",
    }
    headers = {
        "Referer": login_url,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    response = session.post(login_url, data=payload, headers=headers)
    response.raise_for_status()

    # Check if login was successful (redirect to dashboard)
    if response.url != "https://www.pythonanywhere.com/":
        logger.error(f"Login failed. Redirected to {response.url}")
        return False

    logger.info("Login successful.")
    return True


def renew_webapp(session: requests.Session) -> bool:
    """Renew the web app (click the 'Run until 3 months' button)."""
    # First, get the main page to find the renewal URL and CSRF token.
    dashboard_url = "https://www.pythonanywhere.com/"
    logger.info("Getting dashboard...")
    response = session.get(dashboard_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the renewal link/button.
    # Typical HTML: a button with text "Run until 3 months from today"
    # or a form with action containing 'renew'
    # We'll look for a form with action containing '/webapps/' and 'renew'
    renew_url = None
    for form in soup.find_all("form"):
        action = form.get("action", "")
        if "webapps" in action and "renew" in action:
            renew_url = action
            break

    if not renew_url:
        # Alternative: look for a button with the specific text
        for button in soup.find_all("button"):
            if "Run until 3 months from today" in button.get_text():
                parent_form = button.find_parent("form")
                if parent_form:
                    renew_url = parent_form.get("action")
                    break

    if not renew_url:
        logger.error("Could not find renewal form. Is the app already renewed?")
        # Check if there is a message saying "Your web app is running"
        if "Your web app is running" in response.text:
            logger.info("App appears to be already running (maybe already renewed).")
            return True
        return False

    # Make the renewal URL absolute
    if renew_url.startswith("/"):
        renew_url = f"https://www.pythonanywhere.com{renew_url}"

    # Get CSRF token for the renewal form
    logger.info(f"Getting renewal page: {renew_url}")
    response = session.get(renew_url)
    response.raise_for_status()
    csrf_token = get_csrf_token(session, renew_url)
    if not csrf_token:
        logger.error("Failed to get CSRF token for renewal")
        return False

    # Submit the renewal POST
    logger.info("Submitting renewal...")
    payload = {
        "csrfmiddlewaretoken": csrf_token,
        # Usually no other fields needed; sometimes a 'confirm' field
        "confirm": "yes",
    }
    headers = {
        "Referer": renew_url,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    response = session.post(renew_url, data=payload, headers=headers)
    response.raise_for_status()

    # Check for success message
    if "has been renewed" in response.text or "Your web app is running" in response.text:
        logger.info("✅ Renewal successful!")
        return True
    else:
        logger.warning("Renewal may have succeeded, but confirmation message not found.")
        # Still consider it a success if no error
        return True


def main():
    if not USERNAME or not PASSWORD:
        logger.error("PYTHONANYWHERE_USERNAME and PYTHONANYWHERE_PASSWORD must be set.")
        sys.exit(1)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })

    try:
        if not login(session):
            logger.error("Login failed.")
            sys.exit(1)

        if renew_webapp(session):
            logger.info("Web app renewal completed.")
        else:
            logger.error("Renewal failed.")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()