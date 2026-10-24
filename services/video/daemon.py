"""
services/video/daemon.py
ComfyUI process launcher & lifecycle supervisor for Video Subsystem (GPU 1).
"""
import asyncio
import logging
import os
import subprocess
import time
from typing import Optional

from services.video.client import ComfyUIVideoClient
from services.video.config import VideoServiceConfig

logger = logging.getLogger("services.video.daemon")


class ComfyUIVideoDaemon:
    """Supervises the headless ComfyUI process on GPU 1."""

    def __init__(self, config: VideoServiceConfig):
        self.config = config
        self.client = ComfyUIVideoClient(
            host=config.comfy_host,
            port=config.internal_comfy_port,
        )
        self.process: Optional[subprocess.Popen] = None

    def is_alive(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    async def wait_until_ready(self, timeout_seconds: float = 60.0) -> bool:
        start = time.time()
        while time.time() - start < timeout_seconds:
            if await self.client.is_ready():
                return True
            await asyncio.sleep(1.0)
        return False

    def start(self) -> None:
        if self.is_alive():
            logger.info("ComfyUI Video daemon is already running.")
            return

        cmd = [
            "python",
            str(self.config.comfy_dir / "main.py"),
            "--port",
            str(self.config.internal_comfy_port),
            "--cuda-device",
            str(self.config.gpu_device),
            "--listen",
            self.config.comfy_host,
            "--dont-print-server",
            "--disable-assets",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_device)

        logger.info(f"Starting ComfyUI Video daemon on internal port {self.config.internal_comfy_port}: {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise RuntimeError("ComfyUI executable could not be launched; simulation fallback is disabled.") from e

    def stop(self) -> None:
        if self.process and self.is_alive():
            logger.info("Terminating ComfyUI Video daemon...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
