"""
LLM Analyzer - Natural Language Alert Generation
=================================================

PURPOSE:
Uses a pretrained LLM (Ollama/LLaMA) to convert numerical features into
human-readable natural language explanations and alerts.

WHY USE AN LLM?
- Security staff don't want to see "risk_score: 72"
- They want to know "Person #3 is pacing near edge for 8 seconds"
- LLM translates numbers into actionable intelligence

LEARNING CONCEPTS:
- Prompt Engineering: Crafting inputs to get desired outputs
- Few-Shot Learning: LLM understands task from examples in prompt
- API Integration: Calling external services (Ollama)
- Fallback Handling: What to do if LLM fails?

IMPORTANT:
You are NOT training this LLM! You're USING a pretrained model.
Think of it like using Google Translate - you don't train it, you just use it.
"""

import requests
import json
from typing import Dict, Optional
import time


class LLMAnalyzer:
    """
    Generates natural language alerts using pretrained LLM.

    Requirements:
        - Ollama installed and running: `ollama serve`
        - LLaMA model downloaded: `ollama pull llama3.1:8b`

    Example usage:
        analyzer = LLMAnalyzer()

        features = {'min_dist_to_edge': 45, 'dwell_time_near_edge': 8.0, ...}
        risk_score = 72

        result = analyzer.analyze(features, risk_score)
        print(result['alert_message'])
        # "Person #3 at Platform Cam 1: Standing dangerously close to edge for 8 seconds. Immediate intervention recommended."
    """

    def __init__(self,
                 model: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434",
                 timeout: float = 5.0):
        """
        Initialize LLM analyzer.

        Args:
            model: Ollama model name (default: llama3.1:8b)
            base_url: Ollama server URL
            timeout: Max seconds to wait for LLM response

        LEARNING: We configure the LLM connection but don't train anything!
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        print(f"✓ LLMAnalyzer initialized (model={model})")
        print(f"  Ollama URL: {base_url}")

        # Test connection
        if not self._test_connection():
            print("⚠ WARNING: Cannot connect to Ollama!")
            print("  Make sure Ollama is running: `ollama serve`")
            print("  Make sure model is downloaded: `ollama pull llama3.1:8b`")


    def _test_connection(self) -> bool:
        """
        Test if Ollama server is running.

        LEARNING: Always check external services before using them!
        """
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=2.0)
            return response.ok
        except:
            return False


    def analyze(self, features: Dict, risk_score: int, track_id: int = None,
                rag_context: Optional[Dict] = None, camera_id: str = "unknown") -> Dict:
        """
        Generate natural language alert from features and risk score.

        NEW: Now accepts RAG context for context-aware reasoning!

        Args:
            features: Aggregated features from FeatureAggregator
            risk_score: Computed risk score (0-100)
            track_id: Optional person ID for alert message
            rag_context: RAG-retrieved contextual information (NEW!)
                        - train schedules, station info, skip counts, etc.
            camera_id: Camera identifier (for fallback messages)

        Returns:
            Dictionary with:
                - 'risk_level': low/medium/high/critical
                - 'confidence': 0.0-1.0
                - 'reasoning': Why this is flagged (now context-aware!)
                - 'alert_message': Message for security staff
                - 'recommended_action': What to do

        LEARNING: RAG (Retrieval-Augmented Generation) makes LLM smarter!
        Without RAG: "Person near edge"
        With RAG: "Person at Rajiv Chowk let 2 trains pass, train arriving in 30s"
        """
        try:
            # Build prompt for LLM (now with RAG context!)
            prompt = self._build_prompt(features, risk_score, track_id, rag_context)

            # Call Ollama API
            llm_response = self._call_ollama(prompt)

            # Parse JSON response
            result = self._parse_response(llm_response)

            return result

        except Exception as e:
            # If LLM fails, use fallback (rule-based alert)
            print(f"⚠ LLM failed: {e}")
            print("  Using fallback alert")
            return self._fallback_alert(features, risk_score, track_id, camera_id)


    def _build_prompt(self, features: Dict, risk_score: int, track_id: int = None,
                     rag_context: Optional[Dict] = None) -> str:
        """
        Build prompt for LLM with RAG context injection.

        NEW: Now includes RAG context for smarter reasoning!

        LEARNING: RAG (Retrieval-Augmented Generation) workflow:
        1. Retrieve relevant context from database
        2. Inject context into LLM prompt
        3. LLM reasons using both features AND context
        4. Generate context-aware alerts

        This is MUCH better than fine-tuning because:
        - Context can be updated dynamically (no retraining)
        - Works for new stations/schedules immediately
        - Cheaper and faster than fine-tuning
        """
        # System message (role definition)
        system_msg = """You are a metro station safety analyst AI with access to real-time context.
Your role is to analyze behavioral features from CCTV footage and
generate clear, actionable alerts for security staff.

Focus on:
- Platform edge proximity (danger if <50px for >5s)
- Sudden movements toward edge
- Abnormal posture (leaning, torso angle <75° or >105°)
- Erratic pacing patterns (multiple direction changes)
- Prolonged dwelling near dangerous areas
- Phone usage near edge (head_pitch >30° = looking down)
- Sudden acceleration toward edge
- Train arrival timing (critical if person near edge when train approaches)
- Trains skipped (2+ skips = suspicious pattern)

Be concise, clear, and actionable. Consider ALL context when assessing risk.

CRITICAL: You MUST respond with valid JSON only. No markdown, no extra text."""

        # User message (specific task)
        person_id = f"#{track_id}" if track_id else ""

        # Build feature section
        features_section = f"""
PERSON BEHAVIOR DATA:
Person: {person_id}
Rule-based Risk Score: {risk_score}/100
Window Duration: {features.get('window_duration', 0):.1f}s

Movement & Posture:
- Mean torso angle: {features.get('mean_torso_angle', 'N/A')}° (90°=upright, <75°=leaning)
- Mean speed: {features.get('mean_speed', 'N/A')} px/s
- Max speed: {features.get('max_speed', 'N/A')} px/s
- Max acceleration: {features.get('max_acceleration', 'N/A')} px/s² (>300=rushing)
- Acceleration spikes: {features.get('acceleration_spikes', 0)} (>3=erratic)
- Direction changes: {features.get('direction_changes', 0)} (>5=pacing)

Edge Proximity:
- Min distance to platform edge: {features.get('min_dist_to_edge', 'N/A')} px (<50px=danger zone)
- Dwell time near edge: {features.get('dwell_time_near_edge', 0):.1f}s (>5s=suspicious)

Phone Usage Detection (NEW!):
- Mean head pitch: {features.get('mean_head_pitch', 'N/A')}° (>30°=looking down, likely phone)
- Time looking down: {features.get('time_looking_down', 0):.1f}s"""

        # Build RAG context section (if available)
        rag_section = ""
        if rag_context:
            rag_section = f"""

CONTEXTUAL INFORMATION (RAG):
Station & Platform:
- Location: {rag_context.get('station_name', 'Unknown')}
- Platform: {rag_context.get('platform_line', 'Unknown')} Line toward {rag_context.get('platform_direction', 'Unknown')}
- Camera: {rag_context.get('camera_id', 'Unknown')} at {rag_context.get('camera_location', 'Unknown')}

Time Context:
- Current time: {rag_context.get('current_time', 'Unknown')} ({rag_context.get('time_of_day', 'unknown').replace('_', ' ')})
- Day: {rag_context.get('day_of_week', 'Unknown')}
- Peak hour: {'YES - expect crowding' if rag_context.get('is_peak_hour') else 'NO - less crowded'}

Train Information (CRITICAL!):
- Next train arriving in: {rag_context.get('next_train_arrival_seconds', 'Unknown')} seconds
- Train direction: {rag_context.get('next_train_direction', 'Unknown')}
- Frequency: Every {rag_context.get('train_frequency_minutes', 'Unknown')} minutes

Person's History:
- Trains skipped: {rag_context.get('trains_skipped_by_person', 0)} (≥2 = SUSPICIOUS PATTERN!)
- Time on platform: {rag_context.get('person_dwell_time_seconds', 0)} seconds
- Behavior change near train arrival: {'YES - risk increased when train approached!' if rag_context.get('behavior_change_near_arrival') else 'No'}"""

        # Analysis instructions
        instructions = """

ANALYZE THIS SITUATION:
1. Is behavior dangerous given the CONTEXT?
2. If person looking at phone + near edge + train arriving soon → CRITICAL
3. If person skipped 2+ trains + behaving suspiciously → HIGH RISK
4. Consider time of day: peak = crowding normal, off-peak = isolated behavior more suspicious
5. Prioritize train arrival timing - risk is highest when train approaching

Respond with ONLY this JSON format (no markdown):
{
  "risk_level": "low|medium|high|critical",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation using CONTEXT (mention train timing, skipped trains, phone usage if relevant)",
  "alert_message": "Natural language alert for security staff (INCLUDE context: station, train timing, etc.)",
  "recommended_action": "monitor|mic_warning|control_room|driver_alert"
}"""

        # Combine all sections
        full_prompt = f"{system_msg}\n\n{features_section}{rag_section}{instructions}"

        return full_prompt


    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to get LLM response.

        LEARNING: This is where we USE the pretrained LLM (not train it!).
        We send a prompt, LLM processes it, sends back text.

        Args:
            prompt: Text input for LLM

        Returns:
            LLM's generated text response
        """
        url = f"{self.base_url}/api/generate"

        # Request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Get full response at once
            "options": {
                "temperature": 0.3,  # Low temp = more consistent/deterministic
                "top_p": 0.9,        # Nucleus sampling parameter
                "num_predict": 300   # Max tokens to generate
            }
        }

        # Make API call
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()  # Raise error if request failed

        # Extract generated text
        result = response.json()
        generated_text = result.get('response', '')

        return generated_text


    def _parse_response(self, response: str) -> Dict:
        """
        Parse LLM's JSON response.

        LEARNING: LLMs sometimes add extra text or formatting.
        We need to clean it up to extract the JSON.

        Args:
            response: Raw text from LLM

        Returns:
            Parsed dictionary
        """
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]

        response = response.strip()

        # Parse JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"⚠ Failed to parse LLM response as JSON")
            print(f"  Response: {response[:200]}...")
            raise e

        # Validate required fields
        required = ['risk_level', 'confidence', 'reasoning',
                    'alert_message', 'recommended_action']
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        return data


    def _fallback_alert(self, features: Dict, risk_score: int, track_id: int = None, camera_id: str = "unknown") -> Dict:
        """
        Generate simple rule-based alert if LLM fails.

        LEARNING: Always have a backup plan! External services can fail.

        Args:
            features: Behavior features
            risk_score: Risk score
            track_id: Person ID
            camera_id: Camera identifier

        Returns:
            Simple alert dictionary
        """
        person_id = f"#{track_id}" if track_id else "Person"

        # Determine risk level from score
        if risk_score >= 85:
            risk_level = 'critical'
            action = 'driver_alert'
        elif risk_score >= 70:
            risk_level = 'high'
            action = 'control_room'
        elif risk_score >= 50:
            risk_level = 'medium'
            action = 'mic_warning'
        elif risk_score >= 30:
            risk_level = 'low'
            action = 'monitor'
        else:
            risk_level = 'low'
            action = 'monitor'

        # Build simple message
        message_parts = []

        dist = features.get('min_dist_to_edge')
        if dist is not None and dist < 100:
            message_parts.append(f"close to edge ({dist:.0f}px)")

        dwell = features.get('dwell_time_near_edge', 0)
        if dwell > 3:
            message_parts.append(f"dwelling for {dwell:.1f}s")

        speed = features.get('max_speed')
        if speed is not None and speed > 250:
            message_parts.append(f"high speed ({speed:.0f}px/s)")

        changes = features.get('direction_changes', 0)
        if changes > 5:
            message_parts.append(f"pacing ({changes} direction changes)")

        if message_parts:
            details = ", ".join(message_parts)
            alert_message = f"{person_id} detected: {details}. Risk score: {risk_score}/100."
        else:
            alert_message = f"{person_id} detected with risk score {risk_score}/100."

        return {
            'risk_level': risk_level,
            'confidence': 0.5,  # Low confidence for fallback
            'reasoning': 'Automatic rule-based assessment (LLM unavailable)',
            'alert_message': alert_message,
            'recommended_action': action
        }


# EDUCATIONAL EXAMPLE
if __name__ == "__main__":
    """
    Run this file directly to test LLM integration.

    Prerequisites:
    1. Install Ollama: https://ollama.com/download
    2. Pull model: `ollama pull llama3.1:8b`
    3. Start server: `ollama serve`

    Command: python llm_analyzer.py
    """
    print("=== LLM Analyzer Demo ===\n")

    # Create analyzer
    analyzer = LLMAnalyzer()

    # Wait a moment for user to check Ollama
    print("\n⚠ Make sure Ollama is running!")
    print("  Terminal: `ollama serve`")
    input("\nPress Enter when ready...")

    # Test Case: Suspicious Behavior
    print("\n\n--- Test Case: Suspicious Behavior ---")

    features = {
        'track_id': 3,
        'window_duration': 4.2,
        'mean_torso_angle': 72.5,
        'max_torso_angle': 68.0,
        'mean_speed': 180.0,
        'max_speed': 385.0,
        'min_dist_to_edge': 42.0,
        'dwell_time_near_edge': 8.5,
        'direction_changes': 9
    }

    risk_score = 75

    print(f"\nInput Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    print(f"\nRisk Score: {risk_score}/100")

    print("\n⏳ Calling LLM... (may take 2-5 seconds)")
    start_time = time.time()

    result = analyzer.analyze(features, risk_score, track_id=3)

    elapsed = time.time() - start_time

    print(f"\n✓ LLM Response (took {elapsed:.2f}s):")
    print(f"\n  Risk Level: {result['risk_level'].upper()}")
    print(f"  Confidence: {result['confidence']:.2f} ({result['confidence']*100:.0f}%)")
    print(f"\n  Reasoning:")
    print(f"    {result['reasoning']}")
    print(f"\n  Alert Message:")
    print(f"    {result['alert_message']}")
    print(f"\n  Recommended Action: {result['recommended_action']}")

    print("\n\n✓ LLM integration works!")
    print("\nKEY TAKEAWAY:")
    print("The LLM converts raw numbers into human-readable explanations.")
    print("You did NOT train this LLM - you're just USING it like a tool!")
    print("\nFor your hackathon:")
    print("  1. Tell judges: 'I integrated a pretrained LLM for natural language generation'")
    print("  2. Show this demo to demonstrate the LLM generating contextual alerts")
    print("  3. Explain: 'The LLM makes the system more interpretable for security staff'")
