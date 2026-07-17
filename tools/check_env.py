"""Validate environment variables for Lemon Voice Assistant before startup."""

import os
import sys
from pathlib import Path

def main() -> None:
    # Load .env file manually to check values
    env_path = Path(".env")
    if not env_path.exists():
        print("[-] ERROR: .env configuration file not found in current directory!")
        sys.exit(1)

    print("========================================")
    print("      ENVIRONMENT CONFIG VALIDATION     ")
    print("========================================")

    # Simple env parsing
    env_vars = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            # Remove optional quotes
            v = v.strip().strip('"').strip("'")
            env_vars[k.strip()] = v

    errors = 0
    warnings = 0

    # 1. Google Cloud Project Check
    gcp_proj = env_vars.get("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not gcp_proj:
        print("[-] GOOGLE_CLOUD_PROJECT: [MISSING] - CRITICAL: Gemini LLM/STT will fail.")
        errors += 1
    else:
        print(f"[+] GOOGLE_CLOUD_PROJECT: {gcp_proj}")

    # 2. Google Application Credentials Check
    gac = env_vars.get("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if gac:
        gac_path = Path(gac)
        if not gac_path.exists():
            print(f"[-] GOOGLE_APPLICATION_CREDENTIALS: [NOT FOUND] - File '{gac}' does not exist.")
            errors += 1
        else:
            print(f"[+] GOOGLE_APPLICATION_CREDENTIALS: {gac_path.resolve()}")
    else:
        # Check if Application Default Credentials are set up globally
        print("[~] GOOGLE_APPLICATION_CREDENTIALS: Not set. Using global Application Default Credentials.")

    # 3. Web Dashboard Password Check
    web_pass = env_vars.get("VOICE_LOOP_WEB_PASSWORD")
    if not web_pass:
        print("[!] VOICE_LOOP_WEB_PASSWORD: [WARNING] - Not set. Web UI dashboard will be publicly accessible without authentication.")
        warnings += 1
    else:
        print(f"[+] VOICE_LOOP_WEB_PASSWORD: [SET] (Length: {len(web_pass)})")

    # 4. Device Index check
    device_idx_str = env_vars.get("VOICE_LOOP_INPUT_DEVICE_INDEX")
    if device_idx_str is not None:
        try:
            int(device_idx_str)
            print(f"[+] VOICE_LOOP_INPUT_DEVICE_INDEX: {device_idx_str}")
        except ValueError:
            print(f"[-] VOICE_LOOP_INPUT_DEVICE_INDEX: '{device_idx_str}' is not a valid integer index.")
            errors += 1

    print("----------------------------------------")
    print(f"Summary: {errors} Error(s), {warnings} Warning(s).")
    print("========================================\n")

    if errors > 0:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
