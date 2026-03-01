"""
Simple PA System - Pre-recorded Files Only
==========================================

No TTS needed - just plays pre-recorded audio files.
Includes beep generator for driver alerts.
"""

import os
import time
import wave
import struct
import math
from pathlib import Path
from typing import Optional
import threading

# Audio playback
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    USE_PYGAME = True
except ImportError:
    USE_PYGAME = False
    print("⚠ pygame not installed. Run: pip install pygame")


class SimplePASystem:
    """
    Simple PA system that plays pre-recorded audio files
    and generates beep sounds for driver alerts
    """

    def __init__(self, audio_dir: str = "audio_files"):
        """
        Initialize PA system

        Args:
            audio_dir: Directory containing pre-recorded audio files
        """
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(exist_ok=True)

        # Create beep sounds directory
        self.beep_dir = Path("beep_sounds")
        self.beep_dir.mkdir(exist_ok=True)

        # Generate beep sounds if not exists
        self._generate_beep_sounds()

        # Last alert times (for cooldown)
        self.last_alert_time = {}
        self.cooldown_seconds = 5.0

        print(f"✓ PA System initialized")
        print(f"  Audio directory: {self.audio_dir}")
        print(f"  Beep directory: {self.beep_dir}")

    def _generate_beep_sounds(self):
        """Generate beep/buzzer sounds for driver alerts"""

        # Driver alert beep (urgent, loud)
        driver_beep_path = self.beep_dir / "driver_alert.wav"
        if not driver_beep_path.exists():
            print("🔊 Generating driver alert beep...")
            self._create_beep_sound(
                filename=str(driver_beep_path),
                frequency=1000,  # 1000 Hz (urgent tone)
                duration=0.5,    # 0.5 seconds
                repeat=3,        # Repeat 3 times
                gap=0.2          # 0.2s gap between beeps
            )

        # Warning beep (less urgent)
        warning_beep_path = self.beep_dir / "warning_beep.wav"
        if not warning_beep_path.exists():
            print("🔊 Generating warning beep...")
            self._create_beep_sound(
                filename=str(warning_beep_path),
                frequency=800,   # 800 Hz (warning tone)
                duration=0.3,
                repeat=2,
                gap=0.15
            )

        # Emergency buzzer (very urgent, continuous)
        emergency_beep_path = self.beep_dir / "emergency_buzzer.wav"
        if not emergency_beep_path.exists():
            print("🔊 Generating emergency buzzer...")
            self._create_beep_sound(
                filename=str(emergency_beep_path),
                frequency=1200,  # 1200 Hz (emergency tone)
                duration=0.3,
                repeat=5,
                gap=0.1
            )

        print("✓ Beep sounds ready\n")

    def _create_beep_sound(
        self,
        filename: str,
        frequency: int = 1000,
        duration: float = 0.5,
        repeat: int = 1,
        gap: float = 0.2,
        volume: float = 0.8
    ):
        """
        Create a beep sound WAV file

        Args:
            filename: Output filename
            frequency: Beep frequency in Hz
            duration: Duration of each beep in seconds
            repeat: Number of times to repeat
            gap: Gap between repeats in seconds
            volume: Volume (0.0 to 1.0)
        """
        sample_rate = 44100

        # Create beep pattern
        all_samples = []

        for _ in range(repeat):
            # Generate beep
            num_samples = int(sample_rate * duration)
            for i in range(num_samples):
                # Sine wave
                sample = volume * math.sin(2 * math.pi * frequency * i / sample_rate)
                # Envelope (fade in/out to avoid clicks)
                envelope = 1.0
                fade_samples = int(sample_rate * 0.01)  # 10ms fade
                if i < fade_samples:
                    envelope = i / fade_samples
                elif i > num_samples - fade_samples:
                    envelope = (num_samples - i) / fade_samples

                sample *= envelope
                all_samples.append(int(sample * 32767))

            # Add gap
            gap_samples = int(sample_rate * gap)
            all_samples.extend([0] * gap_samples)

        # Write WAV file
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)

            # Pack samples as bytes
            for sample in all_samples:
                wav_file.writeframes(struct.pack('<h', sample))

    def play_audio_file(self, filename: str, async_play: bool = True):
        """
        Play a pre-recorded audio file

        Args:
            filename: Name of audio file (e.g., "level2_english.mp3")
            async_play: Play in background thread (non-blocking)
        """
        audio_path = self.audio_dir / filename

        if not audio_path.exists():
            print(f"⚠ Audio file not found: {audio_path}")
            print(f"  Please add pre-recorded file to: {self.audio_dir}/")
            return

        if async_play:
            threading.Thread(
                target=self._play_audio,
                args=(audio_path,),
                daemon=True
            ).start()
        else:
            self._play_audio(audio_path)

    def _play_audio(self, audio_path: Path):
        """Internal: Play audio file"""
        if not USE_PYGAME:
            print(f"⚠ Cannot play audio - pygame not installed")
            return

        try:
            print(f"🔊 Playing: {audio_path.name}")
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            print(f"✓ Playback completed")
        except Exception as e:
            print(f"✗ Playback failed: {e}")

    def play_driver_alert(self):
        """
        Play driver alert beep (URGENT - for highest risk)
        This is INSTANT - no delay!
        """
        print("🚨 DRIVER ALERT BEEP!")
        beep_path = self.beep_dir / "driver_alert.wav"

        # Play immediately (blocking to ensure driver hears it)
        self._play_audio(beep_path)

    def play_warning_beep(self):
        """Play warning beep (for medium-high risk)"""
        print("⚠️ Warning beep")
        beep_path = self.beep_dir / "warning_beep.wav"
        self._play_audio(beep_path)

    def play_emergency_buzzer(self):
        """Play emergency buzzer (for track intrusion)"""
        print("🚨🚨 EMERGENCY BUZZER!")
        beep_path = self.beep_dir / "emergency_buzzer.wav"
        self._play_audio(beep_path)

    def trigger_alert_by_risk(self, risk_score: float, platform: str = "3", track_id: int = None):
        """
        Trigger appropriate alert based on risk score

        Risk Levels:
        - 0.00-0.39: No alert
        - 0.40-0.79: Warning beep + voice announcement
        - 0.80-0.94: Driver alert beep + voice announcement
        - 0.95+: Emergency buzzer + voice + driver alert

        Args:
            risk_score: Risk score (0-1)
            platform: Platform number
            track_id: Track ID (for cooldown)
        """
        # Check cooldown
        if track_id is not None:
            last_time = self.last_alert_time.get(track_id, 0)
            if time.time() - last_time < self.cooldown_seconds:
                return  # Too soon, skip
            self.last_alert_time[track_id] = time.time()

        if risk_score < 0.40:
            # Level 0 - No alert
            return

        elif risk_score < 0.80:
            # Level 1 - Warning beep + gentle announcement
            print(f"\n⚠️ LEVEL 1 ALERT (Risk: {risk_score:.2f})")
            self.play_warning_beep()
            # Play pre-recorded gentle warning
            self.play_audio_file(f"level1_platform{platform}.mp3", async_play=True)

        elif risk_score < 0.95:
            # Level 2 - Driver alert + security announcement
            print(f"\n🚨 LEVEL 2 ALERT (Risk: {risk_score:.2f})")
            # Play driver beep FIRST (blocking - ensure driver hears it!)
            self.play_driver_alert()
            # Then play PA announcement
            self.play_audio_file(f"level2_platform{platform}.mp3", async_play=True)

        else:
            # Level 3 - EMERGENCY
            print(f"\n🚨🚨 LEVEL 3 EMERGENCY (Risk: {risk_score:.2f})")
            # Play emergency buzzer FIRST
            self.play_emergency_buzzer()
            # Play driver alert
            self.play_driver_alert()
            # Play emergency announcement
            self.play_audio_file(f"level3_platform{platform}.mp3", async_play=True)

    def list_audio_files(self):
        """List all available audio files"""
        print("\n📁 Audio Files:")
        audio_files = list(self.audio_dir.glob("*.*"))
        if not audio_files:
            print(f"  No files found in {self.audio_dir}/")
            print(f"  Add your pre-recorded files there!")
        else:
            for f in audio_files:
                size_kb = f.stat().st_size / 1024
                print(f"  ✓ {f.name} ({size_kb:.1f} KB)")

        print("\n🔊 Beep Sounds:")
        beep_files = list(self.beep_dir.glob("*.wav"))
        for f in beep_files:
            print(f"  ✓ {f.name}")
        print()


# Test the system
if __name__ == "__main__":
    print("=" * 60)
    print("Simple PA System - Test")
    print("=" * 60)

    pa = SimplePASystem()

    print("\n" + "=" * 60)
    print("Available Tests:")
    print("  1. Test warning beep (Level 1)")
    print("  2. Test driver alert beep (Level 2)")
    print("  3. Test emergency buzzer (Level 3)")
    print("  4. Test risk-based alerts (simulated)")
    print("  5. List audio files")
    print("=" * 60)

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        print("\n▶ Testing warning beep...")
        pa.play_warning_beep()

    elif choice == "2":
        print("\n▶ Testing driver alert beep...")
        pa.play_driver_alert()

    elif choice == "3":
        print("\n▶ Testing emergency buzzer...")
        pa.play_emergency_buzzer()

    elif choice == "4":
        print("\n▶ Simulating risk scenarios...\n")

        scenarios = [
            (0.25, "Normal - no alert"),
            (0.65, "Level 1 - Warning"),
            (0.87, "Level 2 - Driver Alert"),
            (0.97, "Level 3 - Emergency")
        ]

        for risk, desc in scenarios:
            print(f"\n{'='*60}")
            print(f"Scenario: {desc} (risk={risk:.2f})")
            print('='*60)
            pa.trigger_alert_by_risk(risk, platform="3", track_id=42)
            time.sleep(3)

    elif choice == "5":
        pa.list_audio_files()

    else:
        print("Invalid choice")

    print("\n✓ Test complete!")
