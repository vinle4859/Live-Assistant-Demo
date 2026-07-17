"""Unit and integration tests for the Operator Web UI & API."""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.request
import os
from pathlib import Path
from unittest.mock import MagicMock

from voice_loop.audio import scale_pcm16_volume
from voice_loop.web_server import start_web_server


def get_free_port() -> int:
    """Find a free TCP port to bind the test server to."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DummyAssistant:
    """Mock LiveVoiceAssistant state interface for HTTP server tests."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.current_mode = "live"
        self._mode_changed_event = asyncio.Event()
        self.mic_gain = 1.0
        self.ambient_phrases = ("Greeting 1", "Greeting 2")
        self.ambient_interval_seconds = 300.0
        self.ambient_enabled = True
        self.next_ambient_time = 0.0
        self.script_lines = []
        self.script_index = 0
        self.script_autoplay = False
        self.loop = loop

        # Track mock method calls
        self.update_script_called = False
        self.script_next_called = False
        self.script_play_all_called = False
        self.script_stop_called = False
        self.ambient_broadcast_called_index = None

    async def update_script_lines_from_raw(self, script_text: str) -> None:
        self.update_script_called = True

    async def trigger_script_next(self) -> None:
        self.script_next_called = True

    async def trigger_script_autoplay(self) -> None:
        self.script_play_all_called = True

    async def trigger_script_stop(self) -> None:
        self.script_stop_called = True

    async def trigger_ambient_broadcast(self, index: int | None = None, text: str | None = None) -> None:
        self.ambient_broadcast_called_index = index

    async def update_ambient_settings(self, phrases: list[str], interval: float, enabled: bool) -> None:
        self.ambient_phrases = tuple(phrases)
        self.ambient_interval_seconds = interval
        self.ambient_enabled = enabled


def test_scale_pcm16_volume() -> None:
    """Verify that pcm16 scaling applies volume gain and clamps overflows correctly."""
    # 2 bytes per sample, 16-bit signed PCM.
    # Samples: [1000, -2000, 30000]
    raw_pcm = b"\xe8\x03\x30\xf8\x30\x75"
    
    # Gain of 1.0 should be a no-op
    assert scale_pcm16_volume(raw_pcm, 1.0) == raw_pcm
    
    # Gain of 2.0: [2000, -4000, 60000 -> clamps to 32767]
    # 2000 = 0x07D0 -> \xd0\x07
    # -4000 = 0xF060 -> \x60\xf0
    # 32767 = 0x7FFF -> \xff\x7f
    expected_pcm = b"\xd0\x07\x60\xf0\xff\x7f"
    assert scale_pcm16_volume(raw_pcm, 2.0) == expected_pcm

    # Clamping negative limit: [-30000] * 2.0 = -60000 -> clamps to -32768
    # -30000 = 0x8AD0 -> \xd0\x8a
    # -32768 = 0x8000 -> \x00\x80
    raw_neg = b"\xd0\x8a"
    expected_neg = b"\x00\x80"
    assert scale_pcm16_volume(raw_neg, 2.0) == expected_neg


def test_web_server_endpoints() -> None:
    """Start the HTTP server on a free local port and test all API operations."""
    # Temporarily clear the password to bypass auth during test
    old_password = os.environ.pop("VOICE_LOOP_WEB_PASSWORD", None)
    try:
        async def run_test():
            loop = asyncio.get_running_loop()
            assistant = DummyAssistant(loop)
            port = get_free_port()
            
            server = start_web_server(assistant, "127.0.0.1", port)
            base_url = f"http://127.0.0.1:{port}"

            try:
                # 1. Test GET /index.html (Static file)
                req = urllib.request.urlopen(f"{base_url}/index.html")
                html_content = req.read().decode("utf-8")
                assert "LEMON OPERATOR DASHBOARD" in html_content
                assert req.status == 200

                # 2. Test GET /api/state
                req = urllib.request.urlopen(f"{base_url}/api/state")
                state = json.loads(req.read().decode("utf-8"))
                assert state["mode"] == "live"
                assert state["mic_gain"] == 1.0
                assert state["ambient_enabled"] is True

                # 3. Test POST /api/mode (Change mode)
                req_data = json.dumps({"mode": "script"}).encode("utf-8")
                request = urllib.request.Request(
                    f"{base_url}/api/mode",
                    data=req_data,
                    headers={"Content-Type": "application/json"}
                )
                response = urllib.request.urlopen(request)
                res_body = json.loads(response.read().decode("utf-8"))
                assert res_body["status"] == "success"
                assert assistant.current_mode == "script"
                await asyncio.sleep(0.1)
                assert assistant._mode_changed_event.is_set()

                # 4. Test POST /api/mic_gain (Change volume scaling)
                req_data = json.dumps({"gain": 1.8}).encode("utf-8")
                request = urllib.request.Request(
                    f"{base_url}/api/mic_gain",
                    data=req_data,
                    headers={"Content-Type": "application/json"}
                )
                response = urllib.request.urlopen(request)
                res_body = json.loads(response.read().decode("utf-8"))
                assert res_body["status"] == "success"
                assert assistant.mic_gain == 1.8

                # 5. Test POST /api/script/update
                req_data = json.dumps({"script_text": "Line 1\nLine 2"}).encode("utf-8")
                request = urllib.request.Request(
                    f"{base_url}/api/script/update",
                    data=req_data,
                    headers={"Content-Type": "application/json"}
                )
                response = urllib.request.urlopen(request)
                res_body = json.loads(response.read().decode("utf-8"))
                assert res_body["status"] == "success"
                # Yield to let main loop execute the async call threadsafe
                await asyncio.sleep(0.1)
                assert assistant.update_script_called is True

                # 6. Test POST /api/script/control (Stepper actions)
                req_data = json.dumps({"action": "next"}).encode("utf-8")
                request = urllib.request.Request(
                    f"{base_url}/api/script/control",
                    data=req_data,
                    headers={"Content-Type": "application/json"}
                )
                response = urllib.request.urlopen(request)
                assert json.loads(response.read().decode("utf-8"))["status"] == "success"
                await asyncio.sleep(0.1)
                assert assistant.script_next_called is True

                # 7. Test POST /api/ambient/settings
                req_data = json.dumps({
                    "phrases": ["Hello Hey Lemon", "Xin Chao Hey Lemon"],
                    "interval": 120.0,
                    "enabled": False
                }).encode("utf-8")
                request = urllib.request.Request(
                    f"{base_url}/api/ambient/settings",
                    data=req_data,
                    headers={"Content-Type": "application/json"}
                )
                response = urllib.request.urlopen(request)
                res_body = json.loads(response.read().decode("utf-8"))
                assert res_body["status"] == "success"
                await asyncio.sleep(0.1)
                assert assistant.ambient_enabled is False
                assert assistant.ambient_interval_seconds == 120.0
                assert "Hello Hey Lemon" in assistant.ambient_phrases

                # 8. Test POST /api/ambient/broadcast_now
                req_data = json.dumps({"index": 1}).encode("utf-8")
                request = urllib.request.Request(
                    f"{base_url}/api/ambient/broadcast_now",
                    data=req_data,
                    headers={"Content-Type": "application/json"}
                )
                response = urllib.request.urlopen(request)
                assert json.loads(response.read().decode("utf-8"))["status"] == "success"
                await asyncio.sleep(0.1)
                assert assistant.ambient_broadcast_called_index == 1

            finally:
                # Shutdown server
                server.shutdown()
                server.server_close()

        asyncio.run(run_test())
    finally:
        if old_password is not None:
            os.environ["VOICE_LOOP_WEB_PASSWORD"] = old_password
