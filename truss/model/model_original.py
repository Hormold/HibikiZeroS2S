# Hibiki Zero S2S — Baseten WebSocket Truss
# Real-time speech-to-speech translation (FR/ES/PT/DE → EN)
#
# WebSocket binary protocol:
#   Server → Client: 0x00                    (handshake)
#   Client → Server: 0x01 + opus_bytes       (input audio)
#   Server → Client: 0x01 + opus_bytes       (translated audio)
#   Server → Client: 0x02 + utf8_text        (translated text)

import asyncio
import random
import time
import logging

import numpy as np
import sentencepiece
import sphn
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

    async def websocket(self, websocket: fastapi.WebSocket):
        """Handle a single WebSocket S2S translation session."""
        async with self._lock:
            opus_writer = sphn.OpusStreamWriter(self._mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self._mimi.sample_rate)

            # Reset streaming state for new session
            self._mimi.reset_streaming()
            self._lm_gen.reset_streaming()

            # Send handshake
            await websocket.send_bytes(b"\x00")
            logger.info("Session started — handshake sent.")

            all_pcm_data = None
            skip_frames = 1
            frame_idx = 0

            try:
                while True:
                    message = await websocket.receive_bytes()

                    if len(message) == 0:
                        continue

                    kind = message[0]

                    if kind != 1:  # only audio supported
                        logger.warning(f"Unknown message kind {kind}")
                        continue

                    # Decode incoming Opus audio
                    payload = message[1:]
                    pcm = opus_reader.append_bytes(payload)
                    if pcm.shape[-1] == 0:
                        continue

                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))

                    # Process complete frames
                    while all_pcm_data.shape[-1] >= self._frame_size:
                        t0 = time.time()
                        chunk = all_pcm_data[: self._frame_size]
                        all_pcm_data = all_pcm_data[self._frame_size :]

                        chunk_tensor = torch.from_numpy(chunk).to(device=DEVICE)[None, None]
                        codes = self._mimi.encode(chunk_tensor)

                        if skip_frames:
                            self._mimi.reset_streaming()
                            skip_frames -= 1

                        for c in range(codes.shape[-1]):
                            tokens = self._lm_gen.step(codes[:, :, c : c + 1])
                            if tokens is None:
                                continue

                            # Decode and send translated audio
                            main_pcm = self._mimi.decode(tokens[:, 1:]).cpu()
                            opus_bytes = opus_writer.append_pcm(
                                main_pcm[0, 0].detach().numpy()
                            )
                            if len(opus_bytes) > 0:
                                await websocket.send_bytes(b"\x01" + opus_bytes)

                            # Send translated text token
                            text_token = tokens[0, 0, 0].item()
                            if text_token not in (0, 3):
                                text = self._text_tokenizer.id_to_piece(text_token)
                                text = text.replace("\u2581", " ")
                                await websocket.send_bytes(
                                    b"\x02" + text.encode("utf-8")
                                )

                        elapsed_ms = 1000 * (time.time() - t0)
                        logger.info(
                            f"frame {frame_idx} processed in {elapsed_ms:.1f}ms"
                        )
                        frame_idx += 1

            except fastapi.WebSocketDisconnect:
                logger.info(f"Session ended after {frame_idx} frames.")
