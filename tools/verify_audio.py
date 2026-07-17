"""Audio device and volume verification utility for live deployment."""

import sys
import os
import subprocess

def get_windows_master_volume() -> tuple[float | None, bool]:
    """Retrieve the Windows master volume level and mute state using inline C# in PowerShell."""
    ps_code = """
$AudioCode = @"
using System;
using System.Runtime.InteropServices;

[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioEndpointVolume {
    int f(); int g(); int h(); int i();
    int SetMasterVolumeLevelScalar(float fLevel, Guid pguidEventContext);
    int j();
    int GetMasterVolumeLevelScalar(out float pfLevel);
    int k(); int l(); int m(); int n();
    int SetMute(bool bMute, Guid pguidEventContext);
    int GetMute(out bool pbMute);
}

[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {
    int Activate(ref Guid id, int clsCtx, int activationParams, out IAudioEndpointVolume aev);
}

[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {
    int f();
    int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice endpoint);
}

[ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
class MMDeviceEnumeratorComObject {}

public class AudioVolume {
    public static string GetVolumeAndMute() {
        try {
            var enumerator = new MMDeviceEnumeratorComObject() as IMMDeviceEnumerator;
            if (enumerator == null) return "ERROR";
            IMMDevice dev = null;
            int hr = enumerator.GetDefaultAudioEndpoint(0, 1, out dev);
            if (hr != 0 || dev == null) return "ERROR_DEVICE";
            IAudioEndpointVolume epv = null;
            Guid epvid = new Guid("5CDF2C82-841E-4546-9722-0CF74078229A");
            hr = dev.Activate(ref epvid, 23, 0, out epv);
            if (hr != 0 || epv == null) return "ERROR_ACTIVATE";
            float v;
            bool mute;
            epv.GetMasterVolumeLevelScalar(out v);
            epv.GetMute(out mute);
            return string.Format("{0:F1}:{1}", v * 100, mute);
        } catch {
            return "ERROR_EXCEPTION";
        }
    }
}
"@

Add-Type -TypeDefinition $AudioCode -ErrorAction SilentlyContinue
[AudioVolume]::GetVolumeAndMute()
"""
    try:
        # Avoid showing PowerShell window or console output popup
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_code],
            capture_output=True,
            text=True,
            timeout=5.0,
            startupinfo=startupinfo
        )
        out = res.stdout.strip()
        if not out or "ERROR" in out:
            return None, False
        
        parts = out.split(":", 1)
        vol = float(parts[0])
        mute = parts[1].strip().lower() == "true"
        return vol, mute
    except Exception:
        return None, False

def verify_devices():
    try:
        import pyaudio
    except ImportError:
        print("[-] PyAudio is not installed. Run 'pip install -r requirements.txt' first.")
        sys.exit(1)

    audio = pyaudio.PyAudio()
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount', 0)
    
    inputs = 0
    outputs = 0
    default_input = None
    default_output = None

    try:
        default_input = audio.get_default_input_device_info()
    except Exception:
        pass

    try:
        default_output = audio.get_default_output_device_info()
    except Exception:
        pass

    print("========================================")
    print("      AUDIO DEVICE DIAGNOSTICS          ")
    print("========================================")

    for i in range(0, numdevices):
        device_info = audio.get_device_info_by_host_api_device_index(0, i)
        name = device_info.get('name')
        max_in = device_info.get('maxInputChannels', 0)
        max_out = device_info.get('maxOutputChannels', 0)
        
        if max_in > 0:
            inputs += 1
            is_def = " (DEFAULT)" if default_input and device_info.get('index') == default_input.get('index') else ""
            print(f"[+] Input Device [{i}]: {name}{is_def}")
        if max_out > 0:
            outputs += 1
            is_def = " (DEFAULT)" if default_output and device_info.get('index') == default_output.get('index') else ""
            print(f"[+] Output Device [{i}]: {name}{is_def}")

    print("----------------------------------------")
    print(f"Summary: Found {inputs} Input Device(s) and {outputs} Output Device(s).")
    
    # Check configured device index from .env if present
    # We do a quick check in .env
    env_input_idx = None
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.startswith("VOICE_LOOP_INPUT_DEVICE_INDEX="):
                try:
                    env_input_idx = int(line.split("=", 1)[1].strip().strip('"').strip("'"))
                except ValueError:
                    pass

    errors = 0
    warnings = 0

    if inputs == 0:
        print("[!] ERROR: No audio Input Devices (Microphone) detected!")
        print("    Ensure your microphone is plugged in and enabled in Windows Settings.")
        errors += 1
    elif env_input_idx is not None:
        # Check if env index exists
        try:
            device_info = audio.get_device_info_by_host_api_device_index(0, env_input_idx)
            if device_info.get('maxInputChannels', 0) == 0:
                print(f"[!] ERROR: Configured input device index [{env_input_idx}] has no input channels!")
                errors += 1
            else:
                print(f"[+] Configured input device index [{env_input_idx}] exists and is valid.")
        except Exception:
            print(f"[!] ERROR: Configured input device index [{env_input_idx}] does not exist in host audio API!")
            errors += 1

    if outputs == 0:
        print("[!] ERROR: No audio Output Devices (Speakers) detected!")
        print("    Ensure your speakers/audio out is connected and enabled.")
        errors += 1

    # Windows Volume & Mute checks
    vol, mute = get_windows_master_volume()
    if vol is not None:
        mute_str = " (MUTED)" if mute else ""
        print(f"[~] Windows Master Playback Volume: {vol:.1f}%{mute_str}")
        if mute:
            print("[!] WARNING: Windows Master Volume is MUTED! Assistant speech will not be audible.")
            warnings += 1
        elif vol < 30.0:
            print(f"[!] WARNING: Windows Master Volume is very low ({vol:.1f}%). Consider turning it up.")
            warnings += 1
    else:
        print("[~] Windows Master Volume: could not be determined natively.")

    print("========================================\n")
    audio.terminate()

    if errors > 0:
        sys.exit(errors) # Returns error code (1, 2 etc.)
    sys.exit(0)

if __name__ == "__main__":
    verify_devices()
