"""Voice edge cases, end to end but offline (no Telegram).

Generates real audio (silence, a mis-tap, pink noise, a rude Russian message,
a Russian offer of 17:30), runs it through the same hear() the agent uses, then
lets the agent answer, and checks the outcome in code. Also checks the agent can
speak its reply in the manager's language.

    python voice_test.py
"""
import asyncio
import os
import re
import sys
import tempfile

import edge_tts

import agent as A
import voice

CASES = {
    # name: (how to make the audio, expected agent behaviour, check on the agent's reply)
    "silence": (("lavfi", "anullsrc=r=48000:cl=mono", 3), "asks to repeat",
                lambda r: r["booking"] is None and re.search(r"(repeat|again|make (it|that) out|type|catch)", " ".join(r["agent_messages"][1:]), re.I)),
    "mis_tap": (("lavfi", "anullsrc=r=48000:cl=mono", 0.4), "asks to repeat",
                lambda r: r["booking"] is None and re.search(r"(repeat|again|make (it|that) out|type|catch)", " ".join(r["agent_messages"][1:]), re.I)),
    "noise": (("lavfi", "anoisesrc=color=pink:amplitude=0.3:r=48000", 4), "asks to repeat",
              lambda r: r["booking"] is None and re.search(r"(repeat|again|make (it|that) out|type|catch)", " ".join(r["agent_messages"][1:]), re.I)),
    "ru_rude": (("tts", "ru-RU-SvetlanaNeural", "Отстань от меня, бот. Мне некогда, не пиши сюда больше."),
                "apologises in Russian and leaves",
                lambda r: r["booking"] is None and r["last_action"] == "give_up"
                and re.search("[а-яА-Я]", r["agent_messages"][-1])),
    "ru_offer": (("tts", "ru-RU-SvetlanaNeural",
                  "Здравствуйте, это Dishoom Shoreditch. На пять часов всё занято, могу предложить полшестого."),
                 "asks in Russian to book 17:30",
                 lambda r: "17:30" in r["agent_messages"][-1] and re.search("[а-яА-Я]", r["agent_messages"][-1])),
}


async def make_audio(spec, path: str) -> None:
    if spec[0] == "lavfi":
        _, src, seconds = spec
        code, out = await voice._run(voice.FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", src,
                                     "-t", str(seconds), "-c:a", "libopus", path)
    else:
        _, name, text = spec
        mp3 = path + ".mp3"
        await edge_tts.Communicate(text, name).save(mp3)
        code, out = await voice._run(voice.FFMPEG, "-y", "-loglevel", "error", "-i", mp3, "-c:a", "libopus", path)
    if code != 0:
        raise RuntimeError(out)


async def main() -> None:
    if sys.platform == "win32":
        os.system("")
        sys.stdout.reconfigure(encoding="utf-8")
    A.LOG_FILE = "decisions_voice_test.log"
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, (spec, expect, ok) in CASES.items():
            print(f"\n\033[1m=== {name} ===\033[0m expect: {expect}")
            path = os.path.join(tmp, name + ".ogg")
            await make_audio(spec, path)
            with open(path, "rb") as f:
                line = await A.hear(f.read())
            result = await A.converse(A.ScriptChannel([line]), A.ScriptOperator([]))
            passed = bool(ok(result))
            results.append((name, passed))
            print("\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m")
            await asyncio.sleep(10)  # free-tier tokens per minute

    print("\n=== reply in the manager's language ===")
    for text in ("Thank you, see you at 17:30.", "Спасибо, до встречи в 17:30.", "Дякую, до зустрічі о 17:30."):
        ogg, seconds = await voice.speak(text)
        heard = await voice.transcribe(A.llm, ogg)
        passed = heard.clear and "17" in heard.text
        results.append((f"speak {voice.language_of(text)}", passed))
        print(f"{'PASS' if passed else 'FAIL'}  {text!r} -> {seconds}s voice -> heard back {heard.text!r}")

    print("\nSUMMARY")
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n{sum(p for _, p in results)}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
