"""Throwaway spike: evaluate Gemini Live (speech-to-speech) for the Hindi agent.

Answers three questions empirically before any S2S re-architecture:
  1. Latency — first audio out, vs our ~3.2s cascade first-word floor.
  2. Hindi/Hinglish voice quality — saved to WAVs to listen (the gating risk).
  3. Control — can a function-call (`record_turn_signal`) carry our action/slots
     alongside the audio?

NOT wired into the pipeline. File-driven: a few Hindi user utterances (generated
via Sarvam TTS as stand-ins for real recordings — note the bias) are streamed
into a Live session and the model's audio reply is captured.

Run (needs network + real Gemini/Sarvam billing):
    .venv/bin/python spikes/gemini_live_spike.py --voice Aoede
    .venv/bin/python spikes/gemini_live_spike.py --model gemini-2.5-flash-preview-native-audio-dialog --voice Aoede
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import os
import re
import statistics as st
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "tmp" / "gemini_live" / "in"
OUT_DIR = ROOT / "tmp" / "gemini_live" / "out"

# Representative Hindi/Hinglish user turns: opening, safety objection,
# code-switched question, closing.
UTTERANCES = [
    ("01_opening", "जी हाँ, बोलिए। क्या बात है?"),
    ("02_safety", "अच्छा, ये ऐप safe है क्या? मुझे थोड़ा डर लग रहा है scam का।"),
    ("03_casino", "अच्छा सुनो, मुझे casino games के बारे में थोड़ा बताओ ना।"),
    ("04_close", "ठीक है ठीक है, link भेज दो WhatsApp पे, मैं देख लूँगा।"),
]

SYSTEM_INSTRUCTION = (
    "You are Anaaya, a warm, energetic, street-smart female sales agent for "
    "'Bharat Matka', an Indian online gaming app. Speak natural, conversational "
    "Hindi the way a real Indian agent does — code-switch to English for brand, "
    "tech and common words (app, link, casino, bonus, WhatsApp) where natural. "
    "Keep every reply short: 1-2 sentences, warm and persuasive. Answer the "
    "customer's question directly first, then gently steer toward sharing the "
    "app link and the joining bonus. Key facts you can use: it's the official, "
    "trusted app; withdrawals are instant; registration is free; 10% cash bonus "
    "on first deposit; start from just 100 rupees; 24x7 WhatsApp support. "
    "Whenever you learn the customer's interest or decide a next step, also call "
    "the record_turn_signal function with the action and any slots you learned."
)


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    vals: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = (p.strip() for p in line.split("=", 1))
        v = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1)) or os.environ.get(m.group(1), ""), v)
        vals[k] = v
    for k, v in vals.items():
        os.environ.setdefault(k, v)


def _read_wav_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, f"{path} must be 16k mono"
        return w.readframes(w.getnframes())


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


async def _gen_inputs() -> None:
    """Synthesize the user utterances via Sarvam TTS (stand-in for real audio)."""
    from src.interfaces.tts import TTSConfig
    from src.providers.tts.sarvam import SarvamTTSAdapter

    key = os.environ.get("TENANT_DEV_SARVAM_KEY") or os.environ.get("SARVAM_API_KEY")
    tts = SarvamTTSAdapter({"api_key": key})
    cfg = TTSConfig(language="hi-IN", voice_id="manisha", sample_rate=16000)  # diff voice from agent
    IN_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in UTTERANCES:
        out = IN_DIR / f"{name}.wav"
        if out.exists():
            continue
        res = await tts.synthesize(text, cfg)
        _write_wav(out, res.audio, 16000)
        print(f"  generated {out.name}  ({len(res.audio)/2/16000:.1f}s)")


async def _run_turn(client, model, config, types, pcm16: bytes) -> dict:
    """One Live session: stream the utterance, capture reply audio + timing."""
    in_tx, out_tx, tool_calls, audio = "", "", [], []
    first_audio_t = None
    async with client.aio.live.connect(model=model, config=config) as session:
        # Stream the utterance in 20ms frames at real-time pace so the model
        # ingests it DURING "speech" (as a live mic would); then measure
        # first-audio from end-of-speech — the realistic turn latency.
        frame = int(16000 * 0.02) * 2  # 20ms @ 16k mono PCM16
        for i in range(0, len(pcm16), frame):
            await session.send_realtime_input(
                audio=types.Blob(data=pcm16[i:i + frame], mime_type="audio/pcm;rate=16000"))
            await asyncio.sleep(0.02)
        await session.send_realtime_input(audio_stream_end=True)
        sent_at = time.monotonic()
        async for msg in session.receive():
            sc = getattr(msg, "server_content", None)
            if sc:
                if sc.input_transcription and sc.input_transcription.text:
                    in_tx += sc.input_transcription.text
                if sc.output_transcription and sc.output_transcription.text:
                    out_tx += sc.output_transcription.text
                if sc.model_turn:
                    for part in sc.model_turn.parts or []:
                        d = getattr(part, "inline_data", None)
                        if d and d.data:
                            if first_audio_t is None:
                                first_audio_t = time.monotonic()
                            audio.append(d.data)
                if sc.turn_complete:
                    break
            tc = getattr(msg, "tool_call", None)
            if tc:
                for fc in tc.function_calls or []:
                    tool_calls.append((fc.name, dict(fc.args or {})))
                    await session.send_tool_response(function_responses=[
                        types.FunctionResponse(id=fc.id, name=fc.name, response={"ok": True})
                    ])
    latency_ms = int((first_audio_t - sent_at) * 1000) if first_audio_t else None
    return {"latency_ms": latency_ms, "in_tx": in_tx, "out_tx": out_tx,
            "tool_calls": tool_calls, "audio24k": b"".join(audio)}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.0-flash-live-001")
    ap.add_argument("--voice", default="Aoede", help="prebuilt Live voice (e.g. Aoede, Kore, Leda, Puck)")
    ap.add_argument("--no-tools", action="store_true", help="run without the function-call tool")
    args = ap.parse_args()
    _load_dotenv()

    gkey = os.environ.get("TENANT_DEV_GEMINI_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gkey:
        print("error: TENANT_DEV_GEMINI_KEY not set"); return 2

    print("=== generating input utterances (Sarvam stand-ins) ===")
    await _gen_inputs()

    from google import genai
    from google.genai import types

    tool = types.Tool(function_declarations=[types.FunctionDeclaration(
        name="record_turn_signal",
        description="Record the dialogue action and any slot values learned this turn.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "action": types.Schema(type="STRING", enum=[
                    "continue", "clarify", "transfer", "schedule_callback",
                    "send_info", "close_positive", "close_negative", "end"]),
                "updated_slots": types.Schema(type="OBJECT"),
            },
            required=["action"],
        ),
    )])
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_INSTRUCTION,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=args.voice)),
            language_code="hi-IN",
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        tools=[] if args.no_tools else [tool],
    )

    client = genai.Client(api_key=gkey)
    print(f"\n=== Gemini Live spike  model={args.model}  voice={args.voice} ===\n")
    lat = []
    for name, _ in UTTERANCES:
        pcm = _read_wav_pcm16(IN_DIR / f"{name}.wav")
        try:
            r = await _run_turn(client, args.model, config, types, pcm)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] ERROR: {type(e).__name__}: {e}")
            continue
        outp = OUT_DIR / f"{name}_{args.voice}.wav"
        if r["audio24k"]:
            pcm16k, _ = audioop.ratecv(r["audio24k"], 2, 1, 24000, 16000, None)
            _write_wav(outp, pcm16k, 16000)
        if r["latency_ms"]:
            lat.append(r["latency_ms"])
        print(f"[{name}] first-audio={r['latency_ms']}ms")
        print(f"   user(asr): {r['in_tx']!r}")
        print(f"   agent    : {r['out_tx']!r}")
        print(f"   tools    : {r['tool_calls']}")
        print(f"   -> {outp.name if r['audio24k'] else '(no audio)'}\n")

    if lat:
        print(f"=== latency (n={len(lat)}): min={min(lat)} median={st.median(lat):.0f} "
              f"max={max(lat)} ms   (cascade baseline first-word ~3200ms) ===")
    print(f"\nAudition:  afplay {OUT_DIR}/*.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
