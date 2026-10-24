"""Run the Factory video queue worker sidecar."""
import asyncio
from services.video.worker import VideoWorkerDaemon

def main() -> None:
    asyncio.run(VideoWorkerDaemon().run())

if __name__ == "__main__":
    main()
