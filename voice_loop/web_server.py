"""HTTP Web Server for the Live Assistant Operator Dashboard."""

from __future__ import annotations

import asyncio
import base64
import http.server
import json
import logging
import os
import socketserver
import threading
from pathlib import Path
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).parent / "web"


class LiveAssistantHTTPHandler(http.server.BaseHTTPRequestHandler):
    """Serve Operator UI static assets and process REST API control requests."""

    # Disable request logging to clean up stdout logs
    def log_message(self, format: str, *args: object) -> None:
        pass

    def _is_authorized(self) -> bool:
        """Check HTTP Basic Authentication headers."""
        password = os.getenv("VOICE_LOOP_WEB_PASSWORD", "").strip()
        if not password:
            return True  # No password configured, bypass authentication

        auth_header = self.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return False

        try:
            encoded_credentials = auth_header.split(" ", 1)[1]
            decoded_bytes = base64.b64decode(encoded_credentials)
            decoded_str = decoded_bytes.decode("utf-8")
            username, pw = decoded_str.split(":", 1)
            return username == "admin" and pw == password
        except Exception:
            return False

    def _send_unauthorized(self) -> None:
        """Send a 401 Unauthorized response with WWW-Authenticate header."""
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Lemon Operator Dashboard"')
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Unauthorized")

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Send a JSON API response."""
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            LOGGER.error("Failed to send JSON response: %s", exc)

    def _read_json(self) -> dict | None:
        """Read and parse JSON from the request body."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                return {}
            raw_body = self.rfile.read(content_length)
            return json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            LOGGER.error("Failed to parse JSON body: %s", exc)
            return None

    def do_OPTIONS(self) -> None:
        """Support CORS preflight options request."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Serve files and return running state."""
        if not self._is_authorized():
            self._send_unauthorized()
            return

        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/api/state":
            assistant = self.server.assistant
            state = {
                "mode": assistant.current_mode,
                "mic_gain": assistant.mic_gain,
                "ambient_enabled": assistant.ambient_enabled,
                "ambient_interval": assistant.ambient_interval_seconds,
                "ambient_phrases": assistant.ambient_phrases,
                "next_ambient_countdown": max(0.0, assistant.next_ambient_time - (assistant.loop.time() if assistant.loop else 0.0))
                if assistant.current_mode == "ambient" and assistant.ambient_enabled and assistant.next_ambient_time > 0
                else 0.0,
                "script_lines": [{"index": line.index, "text": line.text} for line in assistant.script_lines],
                "script_index": assistant.script_index,
                "script_autoplay": assistant.script_autoplay,
            }
            self._send_json(state)
            return

        # Default route to index.html
        if path in {"", "/", "/index.html"}:
            file_path = WEB_ROOT / "index.html"
            content_type = "text/html"
        else:
            self.send_error(404, "File Not Found")
            return

        if not file_path.exists():
            self.send_error(404, f"File {file_path.name} Not Found")
            return

        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:
            LOGGER.error("Failed to serve file %s: %s", file_path, exc)
            self.send_error(500, "Internal Server Error")

    def do_POST(self) -> None:
        """Update settings and execute controller actions."""
        if not self._is_authorized():
            self._send_unauthorized()
            return

        parsed_path = urlparse(self.path)
        path = parsed_path.path
        assistant = self.server.assistant
        loop = assistant.loop

        body = self._read_json()
        if body is None:
            self._send_json({"error": "Invalid JSON body"}, 400)
            return

        if path == "/api/mode":
            mode = body.get("mode")
            if mode not in {"live", "script", "ambient"}:
                self._send_json({"error": f"Invalid mode: {mode}"}, 400)
                return
            LOGGER.info("Operator dashboard request: switch to mode=%s", mode)
            assistant.current_mode = mode
            loop.call_soon_threadsafe(assistant._mode_changed_event.set)
            self._send_json({"status": "success", "mode": assistant.current_mode})
            return

        if path == "/api/mic_gain":
            try:
                gain = float(body.get("gain", 1.0))
                if gain < 0.1 or gain > 5.0:
                    raise ValueError("Gain must be between 0.1 and 5.0")
                assistant.mic_gain = gain
                LOGGER.info("Operator dashboard request: set mic_gain=%.1fx", gain)
                self._send_json({"status": "success", "mic_gain": assistant.mic_gain})
            except Exception as exc:
                self._send_json({"error": str(exc)}, 400)
            return

        if path == "/api/script/update":
            script_text = body.get("script_text", "").strip()
            # Run the rendering pipeline asynchronously on the main loop
            asyncio.run_coroutine_threadsafe(assistant.update_script_lines_from_raw(script_text), loop)
            self._send_json({"status": "success", "message": "Pre-rendering started"})
            return

        if path == "/api/script/control":
            action = body.get("action")
            if action == "next":
                asyncio.run_coroutine_threadsafe(assistant.trigger_script_next(), loop)
            elif action == "play_all":
                if assistant.current_mode != "script":
                    assistant.current_mode = "script"
                    loop.call_soon_threadsafe(assistant._mode_changed_event.set)
                if not assistant.script_autoplay:
                    assistant.script_autoplay = True
                    asyncio.run_coroutine_threadsafe(assistant.trigger_script_autoplay(), loop)
            elif action == "stop":
                assistant.script_autoplay = False
                asyncio.run_coroutine_threadsafe(assistant.trigger_script_stop(), loop)
            elif action == "set_index":
                try:
                    index = int(body.get("index", 0))
                    if 0 <= index <= len(assistant.script_lines):
                        assistant.script_index = index
                        # If playback is active, stop it
                        assistant.script_autoplay = False
                        asyncio.run_coroutine_threadsafe(assistant.trigger_script_stop(), loop)
                    else:
                        raise ValueError("Index out of bounds")
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 400)
                    return
            else:
                self._send_json({"error": f"Invalid action: {action}"}, 400)
                return
            self._send_json({"status": "success"})
            return

        if path == "/api/ambient/settings":
            phrases = body.get("phrases")
            interval = body.get("interval")
            enabled = body.get("enabled")

            p_val = list(phrases) if phrases is not None else list(assistant.ambient_phrases)
            if interval is not None:
                try:
                    i_val = float(interval)
                except ValueError:
                    i_val = assistant.ambient_interval_seconds
            else:
                i_val = assistant.ambient_interval_seconds
            e_val = bool(enabled) if enabled is not None else assistant.ambient_enabled

            LOGGER.info("Operator dashboard request: update ambient settings")
            asyncio.run_coroutine_threadsafe(
                assistant.update_ambient_settings(p_val, i_val, e_val),
                loop
            )
            self._send_json({
                "status": "success",
                "ambient_enabled": e_val,
                "ambient_interval": i_val,
                "ambient_phrases": p_val,
            })
            return

        if path == "/api/ambient/broadcast_now":
            # Force announcement immediately on the main loop
            index = body.get("index")
            text = body.get("text")
            asyncio.run_coroutine_threadsafe(assistant.trigger_ambient_broadcast(index, text), loop)
            self._send_json({"status": "success"})
            return

        self.send_error(404, "Endpoint Not Found")


class LiveAssistantHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Thread-safe HTTP Server tracking a reference to LiveVoiceAssistant."""

    def __init__(self, server_address: tuple[str, int], assistant: object) -> None:
        self.assistant = assistant
        super().__init__(server_address, LiveAssistantHTTPHandler)


def start_web_server(assistant: object, bind_address: str, port: int) -> LiveAssistantHTTPServer:
    """Start the web server in a daemonized background thread."""
    server = LiveAssistantHTTPServer((bind_address, port), assistant)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    LOGGER.info("Web dashboard server started at http://%s:%d", bind_address, port)
    return server
