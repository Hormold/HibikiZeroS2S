"""LiveKit agent: Hibiki Zero real-time speech translation.

Bypasses AgentSession/Agent voice pipeline (turn-based) and connects
participant audio directly to the Baseten-hosted Hibiki model via
WebSocket. Translated audio is published back as a local audio track.

In dev/console mode (no RTC tracks available), falls back to sounddevice
for mic capture and speaker playback.

Usage:
    python agent.py dev          # development mode (mic → Baseten → speakers)
    python agent.py start        # production mode (RTC tracks)
"""

import asyncio
import base64
import json
import logging
import os

import aiohttp
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from livekit import agents, rtc

load_dotenv()

logger = logging.getLogger("hibiki-agent")
logger.setLevel(logging.DEBUG)

# Hibiki model operates at 24 kHz mono PCM16
MODEL_SAMPLE_RATE = 24000
MODEL_CHANNELS = 1

# LiveKit audio defaults
LK_SAMPLE_RATE = 48000
LK_CHANNELS = 1

# Mic chunk size for sounddevice (20ms)
MIC_CHUNK_SAMPLES = LK_SAMPLE_RATE * 20 // 1000  # 960 samples


async def _connect_baseten() -> aiohttp.ClientWebSocketResponse:
    """Open authenticated WebSocket to Baseten Hibiki deployment."""
    ws_url = os.environ["BASETEN_WS_URL"]
    api_key = os.environ["BASETEN_API_KEY"]

    http_session = aiohttp.ClientSession()
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "User-Agent": "LiveKit Hibiki Agent",
    }
    logger.info(f"Connecting to Baseten: {ws_url}")
    ws = await http_session.ws_connect(ws_url, headers=headers)
    logger.info("WebSocket connected, waiting for model handshake (may cold-start)...")

    # Wait for session.created from the model (can take 30-60s on cold start)
    try:
        msg = await asyncio.wait_for(ws.receive(), timeout=120)
        logger.info(f"Received WS msg: type={msg.type}, data_len={len(msg.data) if msg.data else 0}")
        if msg.type == aiohttp.WSMsgType.TEXT:
            ev = json.loads(msg.data)
            logger.info(f"<<< {ev['type']}")
        elif msg.type == aiohttp.WSMsgType.BINARY:
            logger.info(f"<<< binary message, {len(msg.data)} bytes")
        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
            logger.error(f"WebSocket closed during handshake: {msg.type}, extra={msg.extra}")
        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger.error(f"WebSocket error during handshake: {ws.exception()}")
    except asyncio.TimeoutError:
        logger.error("Baseten model did not respond in 120s (cold start?)")
        raise

    # Drain conversation.created if present
    try:
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        if msg.type == aiohttp.WSMsgType.TEXT:
            ev = json.loads(msg.data)
            logger.info(f"<<< {ev['type']}")
    except asyncio.TimeoutError:
        pass

    # Send session config — audio-only, no turn detection
    logger.info("Sending session.update...")
    await ws.send_str(json.dumps({
        "type": "session.update",
        "session": {
            "model": "hibiki-zero-3b",
            "modalities": ["audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": None,
            "input_audio_transcription": None,
        },
    }))

    # Wait for session.updated
    try:
        msg = await asyncio.wait_for(ws.receive(), timeout=10)
        if msg.type == aiohttp.WSMsgType.TEXT:
            ev = json.loads(msg.data)
            logger.info(f"<<< {ev['type']} -- handshake complete")
    except asyncio.TimeoutError:
        logger.warning("No session.updated received, continuing anyway")

    return ws


# ---------------------------------------------------------------------------
# Audio forwarding: participant RTC track → Baseten (production mode)
# ---------------------------------------------------------------------------

async def _forward_participant_audio(
    audio_stream: rtc.AudioStream,
    ws: aiohttp.ClientWebSocketResponse,
    resampler: rtc.AudioResampler,
) -> None:
    """Read participant audio from RTC track, resample to 24kHz, send to Baseten."""
    logger.info("Starting participant RTC audio -> Baseten forwarding")
    async for event in audio_stream:
        frame = event.frame
        for resampled in resampler.push(frame):
            pcm_bytes = resampled.data.tobytes()
            audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
            await ws.send_str(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }))


# ---------------------------------------------------------------------------
# Audio forwarding: local mic → Baseten (dev mode via sounddevice)
# ---------------------------------------------------------------------------

async def _capture_mic_audio(
    ws: aiohttp.ClientWebSocketResponse,
    resampler: rtc.AudioResampler,
) -> None:
    """Capture mic via sounddevice, resample to 24kHz, send to Baseten."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()

    def _sd_callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Mic status: {status}")
        # indata is float32 [-1, 1], convert to int16
        pcm_int16 = (indata[:, 0] * 32767).astype(np.int16)
        loop.call_soon_threadsafe(queue.put_nowait, pcm_int16.tobytes())

    stream = sd.InputStream(
        samplerate=LK_SAMPLE_RATE,
        channels=LK_CHANNELS,
        dtype="float32",
        blocksize=MIC_CHUNK_SAMPLES,
        callback=_sd_callback,
    )

    logger.info(f"Mic capture started (sounddevice, {LK_SAMPLE_RATE}Hz, chunk={MIC_CHUNK_SAMPLES})")
    stream.start()
    try:
        while True:
            pcm_bytes = await queue.get()
            samples = len(pcm_bytes) // 2
            frame = rtc.AudioFrame(
                data=pcm_bytes,
                sample_rate=LK_SAMPLE_RATE,
                num_channels=LK_CHANNELS,
                samples_per_channel=samples,
            )
            for resampled in resampler.push(frame):
                audio_b64 = base64.b64encode(resampled.data.tobytes()).decode("ascii")
                await ws.send_str(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64,
                }))
    finally:
        stream.stop()
        stream.close()


# ---------------------------------------------------------------------------
# Translated audio: Baseten → LiveKit audio source (+ optional speaker queue)
# ---------------------------------------------------------------------------

async def _forward_translated_audio(
    ws: aiohttp.ClientWebSocketResponse,
    audio_source: rtc.AudioSource,
    output_resampler: rtc.AudioResampler,
    speaker_queue: asyncio.Queue[bytes] | None = None,
) -> None:
    """Read translated audio from Baseten, push to LiveKit and optionally speakers."""
    logger.info("Starting Baseten -> output audio forwarding")
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "response.audio.delta":
                pcm_b64 = event.get("delta", "")
                if pcm_b64:
                    pcm_bytes = base64.b64decode(pcm_b64)
                    samples = len(pcm_bytes) // 2  # int16 = 2 bytes
                    if samples == 0:
                        continue

                    frame_24k = rtc.AudioFrame(
                        data=pcm_bytes,
                        sample_rate=MODEL_SAMPLE_RATE,
                        num_channels=MODEL_CHANNELS,
                        samples_per_channel=samples,
                    )

                    for frame_48k in output_resampler.push(frame_24k):
                        await audio_source.capture_frame(frame_48k)
                        if speaker_queue is not None:
                            speaker_queue.put_nowait(frame_48k.data.tobytes())

            elif etype == "error":
                logger.error(f"Baseten error: {event.get('error', {})}")

            elif etype in (
                "response.audio_transcript.delta",
                "response.created",
                "response.done",
            ):
                logger.debug(f"<<< {etype}")

        elif msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.ERROR,
        ):
            logger.warning(f"Baseten WS closed: {msg.type}")
            break


# ---------------------------------------------------------------------------
# Speaker playback (dev mode via sounddevice)
# ---------------------------------------------------------------------------

async def _play_speaker_audio(speaker_queue: asyncio.Queue[bytes]) -> None:
    """Play translated audio through speakers using sounddevice."""
    stream = sd.RawOutputStream(
        samplerate=LK_SAMPLE_RATE,
        channels=LK_CHANNELS,
        dtype="int16",
    )

    logger.info(f"Speaker output started (sounddevice, {LK_SAMPLE_RATE}Hz)")
    stream.start()
    loop = asyncio.get_event_loop()
    try:
        while True:
            pcm_bytes = await speaker_queue.get()
            # RawOutputStream.write() is blocking, run in executor
            await loop.run_in_executor(None, stream.write, pcm_bytes)
    finally:
        stream.stop()
        stream.close()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Resamplers: 48kHz <-> 24kHz
    input_resampler = rtc.AudioResampler(
        input_rate=LK_SAMPLE_RATE,
        output_rate=MODEL_SAMPLE_RATE,
        num_channels=LK_CHANNELS,
    )
    output_resampler = rtc.AudioResampler(
        input_rate=MODEL_SAMPLE_RATE,
        output_rate=LK_SAMPLE_RATE,
        num_channels=MODEL_CHANNELS,
    )

    # Create audio output: 48kHz mono (LiveKit standard)
    audio_source = rtc.AudioSource(LK_SAMPLE_RATE, LK_CHANNELS)
    local_track = rtc.LocalAudioTrack.create_audio_track(
        "hibiki-translation", audio_source
    )
    pub_options = rtc.TrackPublishOptions()
    pub_options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await ctx.room.local_participant.publish_track(local_track, pub_options)
    logger.info("Published translation audio track")

    # Connect to Baseten
    ws = await _connect_baseten()

    # Wait for first human participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Check if participant has audio tracks (real mode) or not (dev mode)
    has_audio_track = any(
        pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO
        for pub in participant.track_publications.values()
    )

    if has_audio_track:
        # --- Production mode: RTC tracks ---
        logger.info("Production mode: using RTC audio tracks")
        recv_task = asyncio.create_task(
            _forward_translated_audio(ws, audio_source, output_resampler)
        )

        forward_task: asyncio.Task | None = None

        def _start_forwarding(track: rtc.Track, who: str) -> None:
            nonlocal forward_task
            logger.info(f"Starting audio forwarding from {who}")
            audio_stream = rtc.AudioStream(
                track, sample_rate=LK_SAMPLE_RATE, num_channels=LK_CHANNELS
            )
            if forward_task and not forward_task.done():
                forward_task.cancel()
            forward_task = asyncio.create_task(
                _forward_participant_audio(audio_stream, ws, input_resampler)
            )

        @ctx.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            remote_participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                _start_forwarding(track, remote_participant.identity)

        for pub in participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                _start_forwarding(pub.track, participant.identity)
                break

        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    else:
        # --- Dev mode: sounddevice mic + speakers ---
        logger.info("Dev mode: no RTC audio tracks, using sounddevice for mic/speakers")
        speaker_queue: asyncio.Queue[bytes] = asyncio.Queue()

        recv_task = asyncio.create_task(
            _forward_translated_audio(ws, audio_source, output_resampler, speaker_queue)
        )
        mic_task = asyncio.create_task(
            _capture_mic_audio(ws, input_resampler)
        )
        play_task = asyncio.create_task(
            _play_speaker_audio(speaker_queue)
        )

        try:
            await asyncio.gather(recv_task, mic_task, play_task)
        except asyncio.CancelledError:
            pass

    await ws.close()
    logger.info("Agent stopped")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
