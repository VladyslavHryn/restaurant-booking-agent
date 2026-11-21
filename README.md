# Booking agent

An AI agent that books a table for two at 17:00 today at a London restaurant by
chatting with the restaurant manager in Telegram. It starts the conversation itself,
in polite British English, reacts to whatever the manager says, never invents facts,
and repeats the confirmed booking (restaurant, date, time, party size, name) once the
manager agrees. It understands voice messages and can reply with a generated British voice.

## How it works

```
 .env (operator brief)                     Telegram (user account, Telethon)
 restaurant, name, time ±flex,      ┌──────────────────────────────────────┐
 phone/notes, voice mode  ───────▶  │ agent.py                             │ ◀── manager's text / voice
                                    │  poll chat ─▶ Whisper (voice→text)   │
                                    │  LLM step: JSON {understood,         │
 console (operator)  ◀── question ──│   decision, action, reply, booking}  │
                     ── answer ───▶ │  code guardrails (state, time window,│ ──▶ text or voice note
                     (90 s timeout) │   recap once, weekday, language)     │     (edge-tts, en-GB)
                                    └──────────────────────────────────────┘
                                                   │
                                                   ▼  one line per message
                                              decisions.log
```

* **Why a user account, not a BotFather bot.** A Bot API bot cannot message a person
  who has not started it first, and the manager never writes first. So the agent runs
  on a regular Telegram account through MTProto (Telethon), logged in once by QR code.
* **The model decides, the code guards.** Every turn the LLM returns JSON: what it
  understood, what it decided, the reply, and the booking on the table. The code owns the
  booking state: it blocks a confirmation outside the brief (time window, party size, date),
  sends the recap exactly once, drops a booking the manager withdraws, fixes a wrong weekday,
  and asks the model to rewrite a recap that misses a detail.
* **No inventing.** The agent knows only the brief in `.env`. Anything else (a deposit,
  a card, an email, a time outside the window) goes to the operator in the console. If the
  operator does not answer in 90 s, the agent tells the manager it does not have that
  information.
* **Decision log.** `decisions.log` gets one line per message, written by the agent:
  what was sent or received, what it understood, what it decided.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # fill in: Telegram api_id/api_hash, Groq key, booking brief
.venv\Scripts\python login.py   # once: scan the QR in Telegram > Settings > Devices
```

Free services only: Telegram API (my.telegram.org), Groq free tier (LLM + Whisper),
edge-tts (no key), ffmpeg bundled via `imageio-ffmpeg`.

## Run

```bash
.venv\Scripts\python agent.py                                  # real chat with MANAGER_USERNAME
.venv\Scripts\python agent.py --simulate scenarios.json        # scripted manager lines, offline
.venv\Scripts\python stress_test.py                            # LLM-played manager personas + checks
```

`stress_test.py` lets a second model play the manager in character (rude, Russian-only,
terse digits, prompt injection, card deposit, details hunter, cancels after confirming,
tomorrow only, earlier slot, emoji only, off-topic small talk, message bursts) and checks
in code: no invented phone/card/email, recap at most once, correct final booking state,
time within the brief, right weekday, reply language, message cap.

## Files

| File | What |
|---|---|
| `agent.py` | the agent: brief, prompt, guardrails, Telegram channel, operator console |
| `voice.py` | text-to-speech (edge-tts → OGG/Opus) and speech-to-text (Whisper on Groq) |
| `login.py` | one-time QR login, saves `agent_session.session` (git-ignored) |
| `scenarios.json` | fixed manager scripts for quick offline checks |
| `stress_test.py` | adaptive LLM manager personas with code-level checks |
| `decisions.log` | decision log of the real conversation |
