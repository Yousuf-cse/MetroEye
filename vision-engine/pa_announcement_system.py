"""
PA Announcement System for MetroEye
===================================

Hybrid approach:
- Level 1 (Gentle): Real-time streaming TTS (~75-300ms latency)
- Level 2-3 (Critical): Pre-cached audio with instant playback
- Flash v2.5 model: ~75ms latency for ultra-low latency
- Turbo v2.5 model: 250-300ms latency

Uses ElevenLabs SDK for streaming audio with fallback to local TTS.
"""

import os
import time
import hashlib
from pathlib import Path
from typing import Optional
import threading

# ElevenLabs SDK for streaming TTS
try:
    from elevenlabs import stream, save
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_SDK_AVAILABLE = True
except ImportError:
    ELEVENLABS_SDK_AVAILABLE = False
    print("⚠ ElevenLabs SDK not installed. Run: pip install elevenlabs")
    print("  Falling back to REST API + local playback")


class PAAnnouncement:
    """
    Manages PA announcements with hybrid approach:
    - Pre-cached for critical alerts (instant playback)
    - Real-time streaming for gentle interventions (75-300ms latency)
    - Uses ElevenLabs SDK for streaming audio playback
    """

    def __init__(
        self,
        elevenlabs_api_key: Optional[str] = None,
        audio_cache_dir: str = "audio_cache",
        use_elevenlabs: bool = True,
        model: str = "eleven_flash_v2_5"
    ):
        """
        Initialize PA announcement system

        Args:
            elevenlabs_api_key: ElevenLabs API key (or set ELEVENLABS_API_KEY env var)
            audio_cache_dir: Directory to cache pre-recorded audio files
            use_elevenlabs: Whether to use ElevenLabs (fallback to local TTS if False)
            model: ElevenLabs model to use:
                   - eleven_flash_v2_5: ~75ms latency (ultra-low latency)
                   - eleven_turbo_v2_5: 250-300ms latency (high quality)
                   - eleven_multilingual_v2: supports 29 languages
        """
        self.api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        self.audio_cache_dir = Path(audio_cache_dir)
        self.audio_cache_dir.mkdir(exist_ok=True)
        self.use_elevenlabs = use_elevenlabs and self.api_key is not None
        self.model = model

        # Initialize ElevenLabs client
        if self.use_elevenlabs and ELEVENLABS_SDK_AVAILABLE:
            self.client = ElevenLabs(api_key=self.api_key)
        else:
            self.client = None

        # ElevenLabs voice IDs (you can customize these)
        # Find more voices at: https://elevenlabs.io/voice-library
        self.voices = {
            "english_male": "pNInz6obpgDQGcFmaJgB",  # Adam (deep, authoritative)
            "english_female": "EXAVITQu4vr4xnSDxMaL",  # Bella (clear, professional)
            "multilingual": "pMsXgVXv3BLzUgSXRplE"  # Multilingual v2 (supports Bengali)
        }

        # Pre-defined critical announcements (Level 2-3)
        # Now supports 3 languages: English, Hindi, Bengali
        self.critical_announcements = {
            "level2_english": {
                "text": "Attention passengers: please stay behind the yellow line at Platform 3. Security assistance requested.",
                "voice": "english_female",
                "language": "en"
            },
            "level2_hindi": {
                "text": "यात्रियों से अनुरोध है: कृपया प्लेटफार्म 3 पर पीली रेखा के पीछे रहें। सुरक्षा सहायता की आवश्यकता है।",
                "voice": "multilingual",
                "language": "hi"
            },
            "level2_bengali": {
                "text": "সকল যাত্রী দয়া করে হলুদ লাইনের পিছনে থাকুন। নিরাপত্তা সহায়তা দরকার।",
                "voice": "multilingual",
                "language": "bn"
            },
            "level3_english": {
                "text": "Emergency alert: immediate evacuation of Platform 3. All passengers move away from the edge now.",
                "voice": "english_male",
                "language": "en"
            },
            "level3_hindi": {
                "text": "आपातकालीन चेतावनी: प्लेटफार्म 3 से तुरंत हटें। सभी यात्री किनारे से दूर हो जाएं।",
                "voice": "multilingual",
                "language": "hi"
            },
            "level3_bengali": {
                "text": "জরুরি সতর্কতা: প্ল্যাটফর্ম ৩ থেকে অবিলম্বে সরে যান। সকলে প্ল্যাটফর্ম ধার থেকে দূরে সরুন।",
                "voice": "multilingual",
                "language": "bn"
            }
        }

        # Pre-generate critical announcements on initialization
        self._pregenerate_critical_announcements()

    def _pregenerate_critical_announcements(self):
        """Pre-generate all critical announcements at startup"""
        print("🔊 Pre-generating critical announcements...")

        for announcement_id, config in self.critical_announcements.items():
            cached_path = self._get_cached_audio_path(config["text"], config["voice"])

            if not cached_path.exists():
                print(f"  Generating: {announcement_id}...")
                try:
                    self._generate_and_cache(
                        text=config["text"],
                        voice_id=self.voices[config["voice"]],
                        language=config.get("language", "en")
                    )
                    print(f"  ✓ {announcement_id} ready")
                except Exception as e:
                    print(f"  ✗ Failed to generate {announcement_id}: {e}")
            else:
                print(f"  ✓ {announcement_id} already cached")

        print("✓ Critical announcements ready\n")

    def _get_cached_audio_path(self, text: str, voice_id: str) -> Path:
        """
        Get cached audio file path based on text and voice

        Args:
            text: Text content
            voice_id: Voice identifier

        Returns:
            Path to cached audio file
        """
        # Create hash of text + voice for unique filename
        content_hash = hashlib.md5(f"{text}_{voice_id}".encode()).hexdigest()[:12]
        return self.audio_cache_dir / f"{content_hash}.mp3"

    def _generate_and_cache(
        self,
        text: str,
        voice_id: str,
        language: str = "en"
    ) -> Path:
        """
        Generate audio using ElevenLabs and cache it

        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            language: Language code (en, bn, etc.)

        Returns:
            Path to generated audio file

        Raises:
            Exception if generation fails
        """
        cached_path = self._get_cached_audio_path(text, voice_id)

        if cached_path.exists():
            return cached_path

        if self.use_elevenlabs:
            try:
                return self._generate_elevenlabs(text, voice_id, cached_path, language)
            except Exception as e:
                print(f"⚠ ElevenLabs failed: {e}. Falling back to local TTS...")
                return self._generate_local_tts(text, cached_path, language)
        else:
            return self._generate_local_tts(text, cached_path, language)

    def _generate_elevenlabs(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        language: str = "en"
    ) -> Path:
        """
        Generate audio using ElevenLabs SDK

        Args:
            text: Text to convert
            voice_id: ElevenLabs voice ID
            output_path: Where to save audio
            language: Language code

        Returns:
            Path to generated audio file
        """
        if not self.client:
            raise Exception("ElevenLabs client not initialized")

        try:
            # Generate audio using SDK
            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=self.model,
                voice_settings={
                    "stability": 0.75,
                    "similarity_boost": 0.85,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            )

            # Save the audio to file
            save(audio_generator, str(output_path))
            return output_path

        except Exception as e:
            raise Exception(f"ElevenLabs SDK error: {e}")

    def _generate_local_tts(
        self,
        text: str,
        output_path: Path,
        language: str = "en"
    ) -> Path:
        """
        Generate audio using local TTS (gTTS fallback)

        Args:
            text: Text to convert
            output_path: Where to save audio
            language: Language code

        Returns:
            Path to generated audio file
        """
        try:
            from gtts import gTTS

            # gTTS supports Bengali with 'bn' code
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(str(output_path))
            return output_path

        except ImportError:
            raise Exception("gTTS not installed. Run: pip install gtts")
        except Exception as e:
            raise Exception(f"Local TTS generation failed: {e}")

    def play_announcement(
        self,
        text: str,
        voice: str = "english_female",
        language: str = "en",
        async_playback: bool = True,
        force_regenerate: bool = False,
        use_streaming: bool = False
    ):
        """
        Play an announcement (generates and caches if needed, or streams in real-time)

        Args:
            text: Text to announce
            voice: Voice identifier (from self.voices)
            language: Language code (en, bn, etc.)
            async_playback: Play in background thread (non-blocking)
            force_regenerate: Force regeneration even if cached
            use_streaming: Stream audio in real-time (lowest latency, no caching)
        """
        voice_id = self.voices.get(voice, self.voices["english_female"])

        # If streaming mode, use real-time TTS (no caching)
        if use_streaming and self.client:
            if async_playback:
                threading.Thread(
                    target=self._play_streaming_audio,
                    args=(text, voice_id),
                    daemon=True
                ).start()
            else:
                self._play_streaming_audio(text, voice_id)
            return

        # Otherwise use cached playback
        cached_path = self._get_cached_audio_path(text, voice_id)

        # Regenerate if forced or not cached
        if force_regenerate or not cached_path.exists():
            try:
                cached_path = self._generate_and_cache(text, voice_id, language)
            except Exception as e:
                print(f"✗ Failed to generate announcement: {e}")
                return

        # Play audio
        if async_playback:
            # Non-blocking - plays in background
            threading.Thread(
                target=self._play_audio_file,
                args=(cached_path,),
                daemon=True
            ).start()
        else:
            # Blocking - waits until playback completes
            self._play_audio_file(cached_path)

    def _play_streaming_audio(self, text: str, voice_id: str):
        """
        Stream and play audio in real-time using ElevenLabs SDK
        Lowest latency approach (75-300ms depending on model)

        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
        """
        try:
            print(f"🔊 Streaming announcement (real-time)...")

            # Get streaming audio generator
            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=self.model,
                voice_settings={
                    "stability": 0.75,
                    "similarity_boost": 0.85,
                    "style": 0.5,
                    "use_speaker_boost": True
                },
                stream=True  # Enable streaming
            )

            # Play the streamed audio directly
            stream(audio_stream)

            print(f"✓ Streaming announcement completed")

        except Exception as e:
            print(f"✗ Streaming playback failed: {e}")
            print(f"  Tip: Ensure elevenlabs SDK is installed: pip install elevenlabs")

    def _play_audio_file(self, audio_path: Path):
        """
        Play cached audio file using system-specific player

        Args:
            audio_path: Path to audio file
        """
        try:
            print(f"🔊 Playing cached announcement: {audio_path.name}")

            # Try using elevenlabs stream on cached file first
            playback_success = False
            if ELEVENLABS_SDK_AVAILABLE:
                try:
                    with open(audio_path, 'rb') as f:
                        audio_data = f.read()
                        # Create an iterator from the audio data
                        def audio_iterator():
                            yield audio_data
                        stream(audio_iterator())
                    playback_success = True
                except Exception as e:
                    # ElevenLabs stream failed (likely mpv not installed)
                    print(f"⚠ ElevenLabs playback failed: {str(e)[:100]}")
                    print(f"  Falling back to Windows Media Player...")

            # Fallback to system player if ElevenLabs failed or not available
            if not playback_success:
                import subprocess
                import platform
                import os as os_module

                system = platform.system()
                if system == "Windows":
                    # Try multiple Windows playback methods
                    try:
                        # Method 1: Use 'start' command (opens with default player)
                        subprocess.run(
                            ["cmd", "/c", "start", "/wait", "", str(audio_path)],
                            check=False,
                            capture_output=True,
                            timeout=30
                        )
                        playback_success = True
                    except:
                        try:
                            # Method 2: Try winsound (Windows built-in)
                            import winsound
                            winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
                            playback_success = True
                        except:
                            try:
                                # Method 3: PowerShell Media.SoundPlayer
                                subprocess.run(
                                    ["powershell", "-c", f"(New-Object Media.SoundPlayer '{audio_path}').PlaySync()"],
                                    check=False,
                                    capture_output=True,
                                    timeout=30
                                )
                                playback_success = True
                            except:
                                print(f"⚠ All Windows playback methods failed")
                                playback_success = False

                elif system == "Darwin":  # macOS
                    subprocess.run(["afplay", str(audio_path)], check=False)
                    playback_success = True
                elif system == "Linux":
                    subprocess.run(["aplay", str(audio_path)], check=False)
                    playback_success = True
                else:
                    print(f"⚠ Unsupported platform: {system}")
                    return

            if playback_success:
                print(f"✓ Announcement completed")
            else:
                print(f"⚠ Playback may have issues")

        except Exception as e:
            print(f"✗ Playback failed: {e}")
            print(f"  Audio file: {audio_path}")
            print(f"  Tip: Install elevenlabs SDK: pip install elevenlabs")

    def play_level1_announcement(self, platform: str = "3"):
        """
        Play Level 1 (Gentle Intervention) announcement in 3 languages
        Uses real-time streaming TTS (75-300ms latency)
        Languages: English → Hindi → Bengali
        """
        english_text = f"Excuse me, please step back from the platform edge at Platform {platform}. Assistance is on the way."
        hindi_text = f"कृपया प्लेटफार्म {platform} के किनारे से पीछे हटें। सहायता आ रही है।"
        bengali_text = "দয়া করে প্ল্যাটফর্ম ধার থেকে পিছু হটুন। সহায়তা আসছে।"

        # Play in sequence: English → Hindi → Bengali
        print("🔊 Level 1: English announcement")
        self.play_announcement(
            english_text,
            voice="english_female",
            language="en",
            async_playback=False,
            use_streaming=True
        )
        time.sleep(0.5)  # Small gap between announcements

        print("🔊 Level 1: Hindi announcement")
        self.play_announcement(
            hindi_text,
            voice="multilingual",
            language="hi",
            async_playback=False,
            use_streaming=True
        )
        time.sleep(0.5)

        print("🔊 Level 1: Bengali announcement")
        self.play_announcement(
            bengali_text,
            voice="multilingual",
            language="bn",
            async_playback=False,
            use_streaming=True
        )

    def play_level2_announcement(self):
        """
        Play Level 2 (Control Room Alert) announcement in 3 languages
        Uses PRE-CACHED audio for instant playback
        Languages: English → Hindi → Bengali
        """
        print("🚨 LEVEL 2 ALERT - Playing critical announcement in 3 languages")

        # These are pre-cached, so playback is INSTANT
        print("🔊 Level 2: English announcement")
        self.play_announcement(
            self.critical_announcements["level2_english"]["text"],
            voice="english_female",
            language="en",
            async_playback=False,
            use_streaming=False  # Use cached version for instant playback
        )
        time.sleep(0.5)

        print("🔊 Level 2: Hindi announcement")
        self.play_announcement(
            self.critical_announcements["level2_hindi"]["text"],
            voice="multilingual",
            language="hi",
            async_playback=False,
            use_streaming=False
        )
        time.sleep(0.5)

        print("🔊 Level 2: Bengali announcement")
        self.play_announcement(
            self.critical_announcements["level2_bengali"]["text"],
            voice="multilingual",
            language="bn",
            async_playback=False,
            use_streaming=False
        )

    def play_level3_announcement(self):
        """
        Play Level 3 (Emergency) announcement in 3 languages
        Uses PRE-CACHED audio for instant playback
        Languages: English → Hindi → Bengali
        """
        print("🚨🚨 LEVEL 3 EMERGENCY - Playing emergency announcement in 3 languages")

        # These are pre-cached, so playback is INSTANT
        print("🔊 Level 3: English announcement")
        self.play_announcement(
            self.critical_announcements["level3_english"]["text"],
            voice="english_male",  # More authoritative voice for emergencies
            language="en",
            async_playback=False,
            use_streaming=False  # Use cached version for instant playback
        )
        time.sleep(0.5)

        print("🔊 Level 3: Hindi announcement")
        self.play_announcement(
            self.critical_announcements["level3_hindi"]["text"],
            voice="multilingual",
            language="hi",
            async_playback=False,
            use_streaming=False
        )
        time.sleep(0.5)

        print("🔊 Level 3: Bengali announcement")
        self.play_announcement(
            self.critical_announcements["level3_bengali"]["text"],
            voice="multilingual",
            language="bn",
            async_playback=False,
            use_streaming=False
        )

    def play_custom_announcement(
        self,
        text: str,
        language: str = "en",
        is_critical: bool = False
    ):
        """
        Play a custom announcement (e.g., generated by Ollama)

        Args:
            text: Announcement text
            language: Language code
            is_critical: If True, uses cached audio; if False, uses streaming
        """
        voice = "english_male" if is_critical else "english_female"

        if is_critical:
            # For critical announcements, use cached version (instant playback)
            self.play_announcement(
                text,
                voice=voice,
                language=language,
                async_playback=False,
                use_streaming=False
            )
        else:
            # For non-critical, use streaming (lowest latency)
            self.play_announcement(
                text,
                voice=voice,
                language=language,
                async_playback=True,
                use_streaming=True
            )

    def clear_cache(self):
        """Clear all cached audio files"""
        for audio_file in self.audio_cache_dir.glob("*.mp3"):
            audio_file.unlink()
        print(f"✓ Cleared {self.audio_cache_dir}")

    def list_cached_files(self):
        """List all cached audio files"""
        files = list(self.audio_cache_dir.glob("*.mp3"))
        print(f"\n📁 Cached audio files ({len(files)}):")
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
        print()


# Example usage and testing
if __name__ == "__main__":
    print("🎙️ PA Announcement System Test\n")
    print("=" * 60)

    # Initialize system (set your API key in .env or pass it here)
    pa_system = PAAnnouncement(
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
        use_elevenlabs=True,  # Set to False to use free gTTS
        model="eleven_flash_v2_5"  # Ultra-low latency (~75ms)
        # model="eleven_turbo_v2_5"  # High quality (250-300ms)
        # model="eleven_multilingual_v2"  # 29 languages
    )

    print("\n" + "=" * 60)
    print("Select announcement to test:")
    print("  1. Level 1 - Gentle Intervention")
    print("  2. Level 2 - Control Room Alert (PRE-RECORDED)")
    print("  3. Level 3 - Emergency (PRE-RECORDED)")
    print("  4. Custom announcement")
    print("  5. List cached files")
    print("  6. Clear cache")
    print("=" * 60)

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == "1":
        print("\n▶ Playing Level 1 announcement...")
        pa_system.play_level1_announcement(platform="3")

    elif choice == "2":
        print("\n▶ Playing Level 2 announcement...")
        pa_system.play_level2_announcement()

    elif choice == "3":
        print("\n▶ Playing Level 3 announcement...")
        pa_system.play_level3_announcement()

    elif choice == "4":
        text = input("Enter announcement text: ")
        lang = input("Language (en/bn): ").strip() or "en"
        is_critical = input("Is critical? (y/n): ").strip().lower() == "y"
        pa_system.play_custom_announcement(text, language=lang, is_critical=is_critical)

    elif choice == "5":
        pa_system.list_cached_files()

    elif choice == "6":
        confirm = input("Clear all cached audio? (y/n): ")
        if confirm.lower() == "y":
            pa_system.clear_cache()

    else:
        print("Invalid choice")

    print("\n✓ Test complete!")
    time.sleep(5)  # Wait for async playback to complete
