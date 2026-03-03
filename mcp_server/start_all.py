"""
3-Server Launcher — Starts Knowledge, CRM, and Automation MCP servers.

Usage:  python -m mcp_server.start_all

Starts:
  Knowledge  → Port 8081 (rag_search)
  CRM        → Port 8082 (get_tasks, get_task_comments, etc.)
  Automation → Port 8083 (create_task, send_notification)

Loads .env and passes all env vars to child processes so they
have access to GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_PROJECT_ID, etc.
"""
import subprocess
import sys
import os
import signal
import logging
import time

# Load .env BEFORE starting subprocesses
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server.launcher")

SERVERS = [
    {"name": "Knowledge",  "module": "mcp_server.knowledge.server",  "port": 8081},
    {"name": "CRM",        "module": "mcp_server.crm.server",        "port": 8082},
    {"name": "Automation", "module": "mcp_server.automation.server", "port": 8083},
]


def main():
    processes = []
    logger.info("=" * 60)
    logger.info("  Starting 3-Server MCP Cluster")
    logger.info("=" * 60)

    # Pass current env (with .env loaded) to child processes
    env = os.environ.copy()

    for srv in SERVERS:
        logger.info(f"  🚀 {srv['name']} → port {srv['port']}")
        proc = subprocess.Popen(
            [sys.executable, "-m", srv["module"]],
            env=env,                     # Pass all env vars including .env
            stdout=sys.stdout,           # Show output in terminal
            stderr=sys.stderr,           # Show errors in terminal
        )
        processes.append((srv["name"], proc))
        time.sleep(1)

        # Check if it crashed immediately
        if proc.poll() is not None:
            logger.error(f"  ❌ {srv['name']} failed to start (exit code: {proc.returncode})")
        else:
            logger.info(f"  ✅ {srv['name']} started (PID: {proc.pid})")

    logger.info("=" * 60)
    logger.info("  All servers launched. Ctrl+C to stop.")
    logger.info("=" * 60)

    def shutdown(signum, frame):
        logger.info("\n🛑 Shutting down...")
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Watchdog: auto-restart crashed servers
    while True:
        for i, (name, proc) in enumerate(processes):
            if proc.poll() is not None:
                logger.error(f"❌ {name} crashed (exit {proc.returncode}). Restarting in 3s...")
                time.sleep(3)
                for srv in SERVERS:
                    if srv["name"] == name:
                        new = subprocess.Popen(
                            [sys.executable, "-m", srv["module"]],
                            env=env,
                            stdout=sys.stdout,
                            stderr=sys.stderr,
                        )
                        processes[i] = (name, new)
                        logger.info(f"  🔄 {name} restarted (PID: {new.pid})")
                        break
        time.sleep(2)


if __name__ == "__main__":
    main()
