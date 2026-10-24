"""Run the Factory image queue worker sidecar."""
import asyncio
from services.image.worker import ImageWorkerDaemon

def main() -> None:
    asyncio.run(ImageWorkerDaemon().run())

if __name__ == "__main__":
    main()
