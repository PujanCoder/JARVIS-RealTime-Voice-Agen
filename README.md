# JARVIS

Personal AI voice assistant.

## Phase 1

Python
    |
FastAPI
    |
Scheduler
    |
Twilio
    |
Phone Call
    |
Voice AI
    |
Two-way Conversation

## Phase 2

Phase 2 adds:

- Gmail
- Google Calendar
- Tasks
- Reminders
- Daily briefing

## Setup

Create virtual environment:

    python -m venv .venv

Activate on Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

Run:

    python -m uvicorn app.main:app --reload
