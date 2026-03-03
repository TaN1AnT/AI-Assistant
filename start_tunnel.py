"""
Start a public tunnel to expose FastAPI (port 8000) to the internet.

Supports multiple tunnel backends:
  1. ngrok (Python package)
  2. localhost.run (via SSH, no install needed)
  3. Manual mode (just prints instructions)

Usage:
    python start_tunnel.py            # tries ngrok, then localhost.run
    python start_tunnel.py --ssh      # forces SSH tunnel (localhost.run)
    python start_tunnel.py --manual   # prints instructions only
"""
import os
import sys
import time
import subprocess
from dotenv import load_dotenv

load_dotenv()

PORT = 8000


def try_ngrok():
    """Try to create tunnel via ngrok Python package."""
    authtoken = os.getenv("NGROK_AUTHTOKEN", "")
    if not authtoken:
        print("  ⚠️  NGROK_AUTHTOKEN not set, skipping ngrok")
        return None

    try:
        import ngrok
        print("  🔄 Connecting via ngrok...")
        listener = ngrok.forward(PORT, authtoken=authtoken)
        return listener.url()
    except Exception as e:
        print(f"  ⚠️  ngrok failed: {e}")
        return None


def try_ssh_tunnel():
    """Create tunnel via localhost.run (SSH-based, no install needed)."""
    print("  🔄 Connecting via localhost.run (SSH)...")
    print("     (First run may ask to accept the SSH host key — type 'yes')\n")

    try:
        proc = subprocess.Popen(
            ["ssh", "-R", f"80:localhost:{PORT}", "nokey@localhost.run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Read output until we find the URL
        for line in proc.stdout:
            line = line.strip()
            if "https://" in line and "localhost.run" in line:
                # Extract URL from the output line
                parts = line.split()
                for part in parts:
                    if part.startswith("https://"):
                        return part, proc
            if "yes/no" in line.lower():
                print("     ❓ SSH is asking to accept the host key.")
                print("     Run this manually first:")
                print(f'     ssh -R 80:localhost:{PORT} nokey@localhost.run')
                proc.kill()
                return None, None

        proc.kill()
        print("  ⚠️  Could not get URL from localhost.run")
        return None, None

    except FileNotFoundError:
        print("  ⚠️  SSH not found. Install OpenSSH or use ngrok.")
        return None, None
    except Exception as e:
        print(f"  ⚠️  SSH tunnel failed: {e}")
        return None, None


def print_config(url):
    """Print n8n configuration details."""
    secret = os.getenv("N8N_WEBHOOK_SECRET", "password")

    print()
    print("=" * 60)
    print(f"  ✅ TUNNEL ACTIVE")
    print("=" * 60)
    print()
    print(f"  📡 Public URL:    {url}")
    print(f"  🔗 Forwards to:  http://localhost:{PORT}")
    print()
    print(f"  ── n8n Configuration ──")
    print(f"  Webhook URL:  {url}/v1/webhook/n8n")
    print(f"  Method:       POST")
    print(f"  Header:       X-Webhook-Secret: {secret}")
    print(f"  Body (JSON):")
    print(f'  {{"message": "Your question here", "session_id": "chat-1"}}')
    print()
    print(f"  ── Test with curl ──")
    print(f'  curl -X POST {url}/v1/webhook/n8n ^')
    print(f'    -H "Content-Type: application/json" ^')
    print(f'    -H "X-Webhook-Secret: {secret}" ^')
    print(f'    -d "{{\\"query\\": \\"test\\"}}"')
    print()
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print("=" * 60)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    print("=" * 60)
    print("  🚀 Starting public tunnel → localhost:8000")
    print("=" * 60)

    if mode == "--manual":
        print()
        print("  Run one of these in a separate terminal:")
        print()
        print(f"  Option 1 (ngrok CLI):")
        print(f"    ngrok http {PORT}")
        print()
        print(f"  Option 2 (SSH, no install):")
        print(f"    ssh -R 80:localhost:{PORT} nokey@localhost.run")
        print()
        print(f"  Option 3 (Cloudflare):")
        print(f"    cloudflared tunnel --url http://localhost:{PORT}")
        return

    # Try ngrok first
    if mode != "--ssh":
        url = try_ngrok()
        if url:
            print_config(url)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n  🛑 Tunnel closed.")
                import ngrok as ng
                ng.disconnect(url)
            return

    # Try SSH tunnel
    result = try_ssh_tunnel()
    if result and result[0]:
        url, proc = result
        print_config(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  🛑 Tunnel closed.")
            if proc:
                proc.kill()
        return

    # Nothing worked
    print()
    print("  ❌ No tunnel method available.")
    print("  Try running manually:")
    print(f"    ssh -R 80:localhost:{PORT} nokey@localhost.run")
    print()


if __name__ == "__main__":
    main()
