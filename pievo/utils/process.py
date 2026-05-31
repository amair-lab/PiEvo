import subprocess
import time
import logging


logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("src.utils.process")


class ServerProcessManager:
    """A context manager to automatically start and stop the server subprocess."""

    def __init__(self, command: list):
        self.command = command
        self.process = None
        logger.info(f"🔧 Preparing to start server with command: {' '.join(command)}")

    def __enter__(self):
        """Start the server process in the background."""
        logger.info("🚀 Starting server subprocess...")
        try:
            # Use Popen to start the process without blocking
            self.process = subprocess.Popen(
                self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            # Give it a moment to initialize
            time.sleep(2)
            logger.warning(f"✅ Server process started with PID: {self.process.pid}")
        except FileNotFoundError:
            logger.error(
                f"❌ Error: The script '{self.command[2]}' was not found. Please check the path."
            )
            raise
        return self.process

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure the server process is terminated upon exiting the block."""
        if (
            self.process and self.process.poll() is None
        ):  # Check if the process is still running
            logger.warning(
                f"🛑 Shutting down server process (PID: {self.process.pid})..."
            )
            try:
                # First, try to terminate gracefully
                self.process.terminate()
                # Wait for up to 5 seconds for it to terminate
                self.process.wait(timeout=5)
                logger.warning("✅ Server process terminated gracefully.")
            except subprocess.TimeoutExpired:
                # If it doesn't terminate, force kill it
                logger.error(
                    "⚠️ Server did not terminate gracefully. Forcing shutdown..."
                )
                self.process.kill()
                self.process.wait()
                logger.error("✅ Server process killed.")
        else:
            logger.error("📦 Server process already finished. ")
