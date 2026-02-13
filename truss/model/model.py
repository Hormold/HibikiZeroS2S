# Hibiki Zero S2S — OpenAI Realtime API Compatible
# Real-time speech-to-speech translation (FR/ES/PT/DE → EN)
#
# Implements the OpenAI Realtime WebSocket JSON protocol so that
# LiveKit Agents (and other compatible clients) can connect directly.
#
# Audio format: PCM16 24kHz mono, base64-encoded in JSON events.
# Supported client events:
#   session.update, input_audio_buffer.append, input_audio_buffer.commit,
#   input_audio_buffer.clear, response.create, response.cancel
# Emitted server events:
#   session.created, session.updated, conversation.created,
#   input_audio_buffer.committed, input_audio_buffer.cleared,
#   conversation.item.created, response.created, response.output_item.added,
#   response.content_part.added, response.audio.delta,
#   response.audio_transcript.delta, response.audio.done,
#   response.audio_transcript.done, response.content_part.done,
#   response.output_item.done, response.done, error

import asyncio
import base64
import json
import logging
import random
import time
import uuid

import numpy as np
import sentencepiece
import torch
import fastapi
from moshi.models import LMGen, LMModel, MimiModel, loaders
from moshi.run_inference import get_condition_tensors

logger = logging.getLogger(__name__)

HF_REPO = "kyutai/hibiki-zero-3b-pytorch-bf16"
HF_REVISION = "23b3e0b41782026c81dd5283a034107b01f9e513"
DEVICE = "cuda"


def seed_all(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def _gen_id(prefix: str = "evt") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Model:
    def __init__(self, *args, **kwargs):
        self._lazy_data_resolver = kwargs["lazy_data_resolver"]

    def load(self):
        if self._lazy_data_resolver:
            self._lazy_data_resolver.block_until_download_complete()

        seed_all(42)
        dtype = torch.float16

        logger.info("Loading checkpoint info from HF...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            HF_REPO, None, None, None, revision=HF_REVISION,
        )

        logger.info("Loading Mimi codec...")
        self._mimi: MimiModel = checkpoint_info.get_mimi(device=DEVICE)

        logger.info("Loading text tokenizer...")
        self._text_tokenizer: sentencepiece.SentencePieceProcessor = (
            checkpoint_info.get_text_tokenizer()
        )

        logger.info("Loading LM (3B)...")
        lm: LMModel = checkpoint_info.get_moshi(device=DEVICE, dtype=dtype)

        logger.info("Initializing LMGen...")
        condition_tensors = get_condition_tensors(
            checkpoint_info.model_type, lm, batch_size=1, cfg_coef=1,
        )
        self._lm_gen = LMGen(
            lm,
            cfg_coef=1,
            condition_tensors=condition_tensors,
            **checkpoint_info.lm_gen_config,
        )

        self._frame_size = int(self._mimi.sample_rate / self._mimi.frame_rate)
        self._lock = asyncio.Lock()

        # Enable streaming mode
        self._mimi.streaming_forever(1)
        self._lm_gen.streaming_forever(1)

        # Warmup
        logger.info("Warming up...")
        self._warmup()
        logger.info("Model ready.")

    def _warmup(self):
        for _ in range(4):
            chunk = torch.zeros(
                1, 1, self._frame_size, dtype=torch.float32, device=DEVICE
            )
            codes = self._mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self._lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self._mimi.decode(tokens[:, 1:])
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # OpenAI Realtime API WebSocket handler
    # ------------------------------------------------------------------

    async def websocket(self, websocket: fastapi.WebSocket):
        """Handle a single WebSocket session using OpenAI Realtime API protocol."""
        async with self._lock:
            self._mimi.reset_streaming()
            self._lm_gen.reset_streaming()

            session_id = _gen_id("sess")
            conv_id = _gen_id("conv")

            session_cfg = {
                "id": session_id,
                "object": "realtime.session",
                "model": "hibiki-zero-3b",
                "modalities": ["text", "audio"],
                "instructions": "",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": None,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200,
                    "create_response": True,
                },
                "tools": [],
                "tool_choice": "auto",
                "temperature": 0.8,
                "max_response_output_tokens": "inf",
            }

            # --- helpers ---------------------------------------------------

            async def send_event(evt: dict):
                evt.setdefault("event_id", _gen_id())
                await websocket.send_json(evt)

            async def send_error(msg: str, code: str = "invalid_request_error",
                                 client_event_id: str | None = None):
                await send_event({
                    "type": "error",
                    "error": {
                        "type": code,
                        "code": code,
                        "message": msg,
                        "param": None,
                        "event_id": client_event_id,
                    },
                })

            # --- response lifecycle ----------------------------------------

            resp_id: str | None = None
            resp_item_id: str | None = None
            transcript_acc: str = ""

            async def begin_response():
                nonlocal resp_id, resp_item_id, transcript_acc
                resp_id = _gen_id("resp")
                resp_item_id = _gen_id("item")
                transcript_acc = ""

                await send_event({
                    "type": "response.created",
                    "response": {
                        "id": resp_id,
                        "object": "realtime.response",
                        "status": "in_progress",
                        "status_details": None,
                        "output": [],
                        "usage": None,
                    },
                })
                await send_event({
                    "type": "response.output_item.added",
                    "response_id": resp_id,
                    "output_index": 0,
                    "item": {
                        "id": resp_item_id,
                        "object": "realtime.item",
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                })
                await send_event({
                    "type": "conversation.item.created",
                    "previous_item_id": None,
                    "item": {
                        "id": resp_item_id,
                        "object": "realtime.item",
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                })
                await send_event({
                    "type": "response.content_part.added",
                    "response_id": resp_id,
                    "item_id": resp_item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "audio", "transcript": ""},
                })

            async def finish_response(status: str = "completed"):
                nonlocal resp_id, resp_item_id
                if resp_id is None:
                    return
                rid, iid = resp_id, resp_item_id
                resp_id = None
                resp_item_id = None

                await send_event({
                    "type": "response.audio.done",
                    "response_id": rid,
                    "item_id": iid,
                    "output_index": 0,
                    "content_index": 0,
                })
                await send_event({
                    "type": "response.audio_transcript.done",
                    "response_id": rid,
                    "item_id": iid,
                    "output_index": 0,
                    "content_index": 0,
                    "transcript": transcript_acc,
                })
                await send_event({
                    "type": "response.content_part.done",
                    "response_id": rid,
                    "item_id": iid,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "audio", "transcript": transcript_acc},
                })
                await send_event({
                    "type": "response.output_item.done",
                    "response_id": rid,
                    "output_index": 0,
                    "item": {
                        "id": iid,
                        "object": "realtime.item",
                        "type": "message",
                        "status": status,
                        "role": "assistant",
                        "content": [
                            {"type": "audio", "transcript": transcript_acc}
                        ],
                    },
                })
                await send_event({
                    "type": "response.done",
                    "response": {
                        "id": rid,
                        "object": "realtime.response",
                        "status": status,
                        "status_details": None,
                        "output": [
                            {
                                "id": iid,
                                "object": "realtime.item",
                                "type": "message",
                                "status": status,
                                "role": "assistant",
                                "content": [
                                    {"type": "audio", "transcript": transcript_acc}
                                ],
                            }
                        ],
                        "usage": {
                            "total_tokens": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "input_token_details": {
                                "text_tokens": 0,
                                "audio_tokens": 0,
                                "cached_tokens": 0,
                            },
                            "output_token_details": {
                                "text_tokens": 0,
                                "audio_tokens": 0,
                            },
                        },
                    },
                })

            # --- audio processing ------------------------------------------

            pcm_buf = np.array([], dtype=np.float32)
            skip_frames = 1
            frame_idx = 0

            async def process_audio():
                """Process buffered PCM through Hibiki, emit response events."""
                nonlocal pcm_buf, skip_frames, frame_idx, transcript_acc

                while pcm_buf.shape[-1] >= self._frame_size:
                    chunk = pcm_buf[: self._frame_size]
                    pcm_buf = pcm_buf[self._frame_size :]

                    t0 = time.time()
                    chunk_tensor = (
                        torch.from_numpy(chunk).to(device=DEVICE)[None, None]
                    )
                    codes = self._mimi.encode(chunk_tensor)

                    if skip_frames:
                        self._mimi.reset_streaming()
                        skip_frames -= 1

                    for c in range(codes.shape[-1]):
                        tokens = self._lm_gen.step(codes[:, :, c : c + 1])
                        if tokens is None:
                            continue

                        # Decode translated audio (float32)
                        main_pcm = self._mimi.decode(tokens[:, 1:]).cpu()
                        pcm_f32 = main_pcm[0, 0].detach().numpy()

                        # Convert float32 → int16 PCM
                        pcm_i16 = (
                            np.clip(pcm_f32 * 32768.0, -32768, 32767)
                            .astype(np.int16)
                        )

                        if pcm_i16.shape[0] > 0:
                            # Auto-start response on first output
                            if resp_id is None:
                                await begin_response()

                            audio_b64 = base64.b64encode(
                                pcm_i16.tobytes()
                            ).decode("ascii")
                            await send_event({
                                "type": "response.audio.delta",
                                "response_id": resp_id,
                                "item_id": resp_item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": audio_b64,
                            })

                        # Text token → transcript delta
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            text = self._text_tokenizer.id_to_piece(
                                text_token
                            )
                            text = text.replace("\u2581", " ")
                            transcript_acc += text

                            if resp_id is not None:
                                await send_event({
                                    "type": "response.audio_transcript.delta",
                                    "response_id": resp_id,
                                    "item_id": resp_item_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": text,
                                })

                    elapsed_ms = 1000 * (time.time() - t0)
                    logger.info(
                        f"frame {frame_idx} processed in {elapsed_ms:.1f}ms"
                    )
                    frame_idx += 1

            # --- initial handshake -----------------------------------------

            await send_event({
                "type": "session.created",
                "session": session_cfg,
            })
            await send_event({
                "type": "conversation.created",
                "conversation": {
                    "id": conv_id,
                    "object": "realtime.conversation",
                },
            })
            logger.info("Session started — session.created sent.")

            # --- main event loop -------------------------------------------

            try:
                while True:
                    raw = await websocket.receive_text()

                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        await send_error("Failed to parse JSON", "invalid_json")
                        continue

                    etype = event.get("type")
                    client_eid = event.get("event_id")

                    # ---- session.update ------------------------------------
                    if etype == "session.update":
                        s = event.get("session", {})
                        for key in (
                            "modalities", "instructions", "voice",
                            "input_audio_format", "output_audio_format",
                            "input_audio_transcription", "turn_detection",
                            "temperature", "max_response_output_tokens",
                            "tools", "tool_choice",
                        ):
                            if key in s:
                                session_cfg[key] = s[key]
                        await send_event({
                            "type": "session.updated",
                            "session": session_cfg,
                        })

                    # ---- input_audio_buffer.append -------------------------
                    elif etype == "input_audio_buffer.append":
                        audio_b64 = event.get("audio", "")
                        if audio_b64:
                            raw_bytes = base64.b64decode(audio_b64)
                            pcm_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
                            pcm_f32 = pcm_i16.astype(np.float32) / 32768.0
                            pcm_buf = np.concatenate([pcm_buf, pcm_f32])
                            await process_audio()

                    # ---- input_audio_buffer.commit -------------------------
                    elif etype == "input_audio_buffer.commit":
                        await process_audio()
                        await finish_response()

                        user_item_id = _gen_id("item")
                        await send_event({
                            "type": "input_audio_buffer.committed",
                            "previous_item_id": None,
                            "item_id": user_item_id,
                        })
                        await send_event({
                            "type": "conversation.item.created",
                            "previous_item_id": None,
                            "item": {
                                "id": user_item_id,
                                "object": "realtime.item",
                                "type": "message",
                                "status": "completed",
                                "role": "user",
                                "content": [
                                    {"type": "input_audio", "transcript": None}
                                ],
                            },
                        })

                    # ---- input_audio_buffer.clear --------------------------
                    elif etype == "input_audio_buffer.clear":
                        pcm_buf = np.array([], dtype=np.float32)
                        await send_event({
                            "type": "input_audio_buffer.cleared",
                        })

                    # ---- response.create -----------------------------------
                    elif etype == "response.create":
                        await finish_response()
                        # Future audio will auto-start a new response

                    # ---- response.cancel -----------------------------------
                    elif etype == "response.cancel":
                        await finish_response("cancelled")

                    # ---- conversation.item.* (ack but no-op) ---------------
                    elif etype in (
                        "conversation.item.create",
                        "conversation.item.truncate",
                        "conversation.item.delete",
                    ):
                        pass  # translation model ignores conversation ops

                    else:
                        if etype:
                            logger.debug(f"Ignoring unknown event: {etype}")

            except fastapi.WebSocketDisconnect:
                try:
                    await finish_response()
                except Exception:
                    pass
                logger.info(f"Session ended after {frame_idx} frames.")
