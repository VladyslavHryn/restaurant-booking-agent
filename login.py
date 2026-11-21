"""One-time Telegram login by QR code (no phone number or SMS code typing).

Run once:  python login.py
Then on your phone: Telegram -> Settings -> Devices -> Link Desktop Device,
and scan the QR code shown in the terminal. The session is saved to
<TG_SESSION>.session and agent.py reuses it from then on.
"""
import asyncio
import getpass
import os
import sys

import qrcode
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

load_dotenv()
API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
SESSION = os.getenv("TG_SESSION", "agent_session")


def show_qr(url: str) -> None:
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.print_ascii(invert=True)


async def main() -> None:
    client = TelegramClient(SESSION, API_ID, API_HASH)
    await client.connect()
    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"Already logged in as @{me.username or me.first_name}. Nothing to do.")
        return

    qr_login = await client.qr_login()
    while True:
        print("\nScan with Telegram on your phone: Settings -> Devices -> Link Desktop Device\n")
        show_qr(qr_login.url)
        try:
            await qr_login.wait(timeout=60)
            break
        except asyncio.TimeoutError:
            print("QR expired, generating a new one...")
            await qr_login.recreate()
        except SessionPasswordNeededError:
            pwd = getpass.getpass("Two-step verification password: ")
            await client.sign_in(password=pwd)
            break

    me = await client.get_me()
    print(f"\nLogged in as @{me.username or me.first_name}. Session saved to {SESSION}.session")
    await client.disconnect()


if __name__ == "__main__":
    if sys.platform == "win32":
        os.system("")  # enable ANSI/Unicode block rendering in old consoles
    asyncio.run(main())
