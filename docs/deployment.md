# Deployment Guide

This guide covers deploying the Telegram bot to **PythonAnywhere** (free tier). The bot runs as a Flask web application with a webhook, with background services (health checker, analytics) running in threads.

---

## Prerequisites

- A [PythonAnywhere](https://www.pythonanywhere.com/) account (free tier is sufficient).
- Git repository with your bot code (GitHub, GitLab, or direct upload).
- Your Telegram bot token from [@BotFather](https://t.me/botfather).
- Your OpenRouter API key.

---

## Step 1: Set Up the Web App

1. **Log in** to PythonAnywhere.
2. Go to the **Web** tab.
3. Click **Add a new web app**.
4. Choose **Manual configuration** (not Flask or Django templates).
5. Select your Python version (3.10 or newer).
6. Set the **Source code** path to your project directory, e.g.: