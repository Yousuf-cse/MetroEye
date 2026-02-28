"""
Asynchronous Alert Generation System
=====================================

Two-stage architecture for fast detection + detailed descriptions:

Stage 1: Real-time XGBoost classification (0.1ms) - BLOCKING
Stage 2: Ollama description generation (2-5s) - NON-BLOCKING (async)

This ensures:
- Detection loop runs at 30+ FPS (no slowdown)
- Detailed descriptions generated in background
- Immediate alarms for dangerous situations
"""

import time
import json
import queue
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import requests


class AsyncAlertSystem:
    """
    Manages asynchronous alert generation and delivery

    Flow:
    1. Detection loop detects high risk → queue alert
    2. Background worker picks up alert → calls Ollama
    3. Ollama generates description (2-5s, doesn't block detection)
    4. Send to backend API + trigger voice announcement
    """

    def __init__(
        self,
        ollama_url="http://localhost:11434",
        backend_url="http://localhost:8000/api/alerts",
        alert_cooldown=10.0
    ):
        """
        Initialize async alert system

        Args:
            ollama_url: Ollama API endpoint
            backend_url: Backend API endpoint for alerts
            alert_cooldown: Minimum seconds between alerts for same person
        """
        self.ollama_url = ollama_url
        self.backend_url = backend_url
        self.alert_cooldown = alert_cooldown

        # Alert queue (detection loop adds to this)
        self.alert_queue = queue.Queue(maxsize=100)

        # Track recent alerts to avoid spam
        self.recent_alerts = {}  # track_id -> last_alert_timestamp

        # Background worker thread
        self.worker_thread = None
        self.running = False

    def start(self):
        """Start background worker thread"""
        if self.running:
            print("⚠ Alert system already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("✓ Async alert system started")

    def stop(self):
        """Stop background worker thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        print("✓ Async alert system stopped")

    def queue_alert(
        self,
        track_id: int,
        risk_score: float,
        confidence: float,
        features: Dict[str, Any],
        frame: Any,
        bbox: tuple,
        appearance: Dict[str, Any],
        camera_id: str,
        timestamp: float
    ):
        """
        Queue an alert for async processing

        Called from detection loop when high risk detected.
        This is FAST (just adds to queue, doesn't block).

        Args:
            track_id: Person track ID
            risk_score: Risk score from XGBoost (0-1)
            confidence: Model confidence
            features: Behavioral features dict
            frame: Video frame (for context)
            bbox: Person bounding box
            appearance: Person appearance dict
            camera_id: Camera identifier
            timestamp: Detection timestamp
        """
        # Check cooldown
        last_alert_time = self.recent_alerts.get(track_id, 0)
        if (timestamp - last_alert_time) < self.alert_cooldown:
            return  # Too soon, skip

        # Update cooldown
        self.recent_alerts[track_id] = timestamp

        # Create alert data
        alert_data = {
            'track_id': track_id,
            'risk_score': risk_score,
            'confidence': confidence,
            'features': features,
            'frame': frame,
            'bbox': bbox,
            'appearance': appearance,
            'camera_id': camera_id,
            'timestamp': timestamp
        }

        # Add to queue (non-blocking)
        try:
            self.alert_queue.put_nowait(alert_data)
            print(f"📬 Queued alert for track {track_id} (risk: {risk_score:.2f})")
        except queue.Full:
            print(f"⚠ Alert queue full, dropping alert for track {track_id}")

    def _worker_loop(self):
        """
        Background worker thread

        Continuously processes alerts from queue.
        This runs in background, doesn't block detection loop.
        """
        print("🔄 Alert worker thread started")

        while self.running:
            try:
                # Wait for alert (blocks here, not in detection loop!)
                alert_data = self.alert_queue.get(timeout=1.0)

                # Process alert (THIS IS SLOW, 2-5 seconds)
                self._process_alert(alert_data)

                # Mark as done
                self.alert_queue.task_done()

            except queue.Empty:
                # No alerts to process, continue waiting
                continue
            except Exception as e:
                print(f"✗ Error processing alert: {e}")

        print("🔄 Alert worker thread stopped")

    def _process_alert(self, alert_data: Dict[str, Any]):
        """
        Process a single alert (SLOW - runs in background)

        Steps:
        1. Generate description with Ollama (2-5 seconds)
        2. Create JSON payload
        3. Send to backend API
        4. Trigger voice announcement (optional)

        Args:
            alert_data: Alert data dict
        """
        start_time = time.time()

        track_id = alert_data['track_id']
        risk_score = alert_data['risk_score']

        print(f"🤖 Processing alert for track {track_id}...")

        # Generate risk level
        risk_level = self._get_risk_level(risk_score)

        # Generate description with Ollama (SLOW PART)
        try:
            description = self._generate_ollama_description(alert_data)
        except Exception as e:
            print(f"⚠ Ollama generation failed: {e}")
            description = self._generate_fallback_description(alert_data)

        # Create JSON payload
        payload = self._create_alert_payload(alert_data, description, risk_level)

        # Send to backend API
        self._send_to_backend(payload)

        # Generate voice announcement (optional)
        if risk_level in ['critical', 'high']:
            self._generate_voice_announcement(description.get('voice_announcement', ''))

        elapsed = time.time() - start_time
        print(f"✓ Alert processed in {elapsed:.2f}s (track {track_id})")

    def _generate_ollama_description(self, alert_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate alert description using Ollama (2-5 seconds)

        Args:
            alert_data: Alert data

        Returns:
            dict: Generated descriptions
        """
        features = alert_data['features']
        appearance = alert_data['appearance']
        risk_score = alert_data['risk_score']

        # Create prompt for Ollama
        prompt = f"""You are a safety monitoring AI. Generate a concise alert for security staff.

Context:
- Person: {appearance.get('description', 'Person')}
- Risk Score: {risk_score:.2f} (0=safe, 1=extreme danger)
- Distance from edge: {features.get('dist_to_edge', 'unknown')}px (~{features.get('dist_to_edge', 0) / 100:.1f} meters)
- Dwell time at edge: {features.get('dwell_time', 0):.1f} seconds
- Body angle: {features.get('torso_angle', 'unknown')} degrees
- Movement speed: {features.get('speed', 'unknown')} px/s

Generate:
1. alert_message: One sentence critical alert for staff (include person appearance)
2. recommended_action: What staff should do immediately
3. voice_announcement: Brief spoken announcement for PA system (20-30 words max)

Format as JSON with those three keys. Be concise and urgent."""

        # Call Ollama API
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": "llama3.2:3b",  # Use fast model
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Low temperature for consistency
                    "max_tokens": 200
                }
            },
            timeout=10.0
        )

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')

            # Parse JSON from response
            try:
                # Extract JSON from markdown code blocks if present
                if '```json' in generated_text:
                    json_start = generated_text.index('```json') + 7
                    json_end = generated_text.index('```', json_start)
                    generated_text = generated_text[json_start:json_end].strip()
                elif '```' in generated_text:
                    json_start = generated_text.index('```') + 3
                    json_end = generated_text.index('```', json_start)
                    generated_text = generated_text[json_start:json_end].strip()

                descriptions = json.loads(generated_text)
                return descriptions
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠ Failed to parse Ollama response: {e}")
                return self._generate_fallback_description(alert_data)
        else:
            raise Exception(f"Ollama API error: {response.status_code}")

    def _generate_fallback_description(self, alert_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate simple description without Ollama (fallback)

        Args:
            alert_data: Alert data

        Returns:
            dict: Basic descriptions
        """
        appearance = alert_data['appearance']
        features = alert_data['features']
        risk_score = alert_data['risk_score']
        track_id = alert_data['track_id']

        person_desc = appearance.get('description', 'Person')
        dist_edge = features.get('dist_to_edge', 999)
        dwell_time = features.get('dwell_time', 0)

        return {
            "alert_message": f"⚠️ HIGH RISK: {person_desc} at platform edge ({dist_edge:.0f}px, {dwell_time:.1f}s). Risk score: {risk_score:.2f}",
            "recommended_action": f"Dispatch security to Track {track_id} immediately. Person is dangerously close to platform edge.",
            "voice_announcement": f"Attention: Emergency at Track {track_id}. {person_desc} requires immediate assistance."
        }

    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to level"""
        if risk_score >= 0.9:
            return "critical"
        elif risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.5:
            return "medium"
        elif risk_score >= 0.3:
            return "low"
        else:
            return "normal"

    def _create_alert_payload(
        self,
        alert_data: Dict[str, Any],
        description: Dict[str, str],
        risk_level: str
    ) -> Dict[str, Any]:
        """
        Create final JSON payload for backend API

        Args:
            alert_data: Alert data
            description: Generated descriptions
            risk_level: Risk level string

        Returns:
            dict: Complete alert payload
        """
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_track{alert_data['track_id']}"

        return {
            "alert_id": alert_id,
            "timestamp": datetime.fromtimestamp(alert_data['timestamp']).isoformat() + 'Z',
            "camera_id": alert_data['camera_id'],
            "track_id": alert_data['track_id'],
            "risk_level": risk_level,
            "risk_score": round(alert_data['risk_score'], 3),
            "confidence": round(alert_data['confidence'], 3),

            "person_appearance": {
                "description": alert_data['appearance'].get('description', 'Unknown'),
                "upper_clothing": alert_data['appearance'].get('upper_clothing', 'unknown'),
                "lower_clothing": alert_data['appearance'].get('lower_clothing', 'unknown'),
                "height_estimate": alert_data['appearance'].get('height', 'unknown'),
                "distinguishing_features": "carrying a backpack" if alert_data['appearance'].get('has_bag') else ""
            },

            "behavior_analysis": {
                "primary_concern": description.get('alert_message', 'Unknown concern'),
                "duration": f"{alert_data['features'].get('dwell_time', 0):.1f} seconds",
                "distance_from_edge": f"{alert_data['features'].get('dist_to_edge', 0):.0f} pixels",
                "body_posture": f"Torso angle: {alert_data['features'].get('torso_angle', 0):.0f} degrees",
                "movement_pattern": self._describe_movement(alert_data['features'])
            },

            "alert_message": description.get('alert_message', ''),
            "recommended_action": description.get('recommended_action', ''),
            "voice_announcement": description.get('voice_announcement', ''),

            "location": {
                "bbox": list(alert_data['bbox']),
                "platform_zone": "danger_zone" if alert_data['features'].get('dist_to_edge', 999) < 50 else "caution_zone"
            }
        }

    def _describe_movement(self, features: Dict[str, Any]) -> str:
        """Describe movement pattern"""
        speed = features.get('speed', 0)

        if speed < 5:
            return "Stationary, not moving away from edge"
        elif speed < 20:
            return "Slow movement near edge"
        else:
            return "Moving along platform"

    def _send_to_backend(self, payload: Dict[str, Any]):
        """
        Send alert to backend API

        Args:
            payload: Alert JSON payload
        """
        try:
            response = requests.post(
                self.backend_url,
                json=payload,
                timeout=5.0
            )

            if response.status_code == 200:
                print(f"✓ Alert sent to backend: {payload['alert_id']}")
            else:
                print(f"⚠ Backend returned {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to send to backend: {e}")
            # Log to file as fallback
            self._log_alert_to_file(payload)

    def _log_alert_to_file(self, payload: Dict[str, Any]):
        """Log alert to file if backend unavailable"""
        with open('alerts_log.jsonl', 'a') as f:
            f.write(json.dumps(payload) + '\n')
        print(f"✓ Alert logged to file: {payload['alert_id']}")

    def _generate_voice_announcement(self, text: str):
        """
        Generate voice announcement (optional)

        You can integrate with:
        - TTS engines (pyttsx3, gTTS)
        - External APIs (Google Cloud TTS, AWS Polly)
        - PA system integration

        Args:
            text: Text to speak
        """
        # TODO: Implement TTS if needed
        print(f"🔊 Voice announcement: {text}")
        # Example:
        # import pyttsx3
        # engine = pyttsx3.init()
        # engine.say(text)
        # engine.runAndWait()


# Example usage
if __name__ == "__main__":
    # Initialize alert system
    alert_system = AsyncAlertSystem(
        ollama_url="http://localhost:11434",
        backend_url="http://localhost:8000/api/alerts",
        alert_cooldown=10.0
    )

    # Start background worker
    alert_system.start()

    # Simulate detection loop
    print("\n🎬 Simulating detection loop...")

    for i in range(5):
        print(f"\n[Frame {i}] Detection running at 30 FPS (no slowdown)...")

        # Simulate high risk detection
        if i == 2:  # Risk detected at frame 2
            print("⚠️ HIGH RISK DETECTED!")

            # Queue alert (FAST - just adds to queue)
            alert_system.queue_alert(
                track_id=47,
                risk_score=0.92,
                confidence=0.88,
                features={
                    'dist_to_edge': 18,
                    'dwell_time': 12.3,
                    'torso_angle': 45,
                    'speed': 2.1
                },
                frame=None,  # Would be actual frame
                bbox=(523, 245, 687, 512),
                appearance={
                    'description': 'Tall person wearing dark blue jacket and black pants',
                    'upper_clothing': 'blue',
                    'lower_clothing': 'black',
                    'height': 'tall',
                    'has_bag': True
                },
                camera_id='platform_cam_1',
                timestamp=time.time()
            )

        time.sleep(0.033)  # 30 FPS = 33ms per frame

    print("\n✓ Detection loop complete. Alert processing in background...")

    # Wait for alerts to process
    time.sleep(8)

    # Stop alert system
    alert_system.stop()

    print("\n✓ Demo complete!")
