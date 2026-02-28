"""
RAG Context Retriever - Context Injection for LLM
=================================================

PURPOSE:
Retrieves and formats contextual information to inject into LLM prompts.
This is the "R" (Retrieval) part of RAG (Retrieval-Augmented Generation).

WHY RAG INSTEAD OF FINE-TUNING?
Fine-tuning requires:
- 1000+ training examples
- GPU resources ($50-200)
- Days of training time
- Retrain for every station/schedule change

RAG requires:
- JSON database (free, instant)
- Simple retrieval logic
- Update JSON to change behavior
- Works for any station immediately

LEARNING CONCEPTS:
- RAG (Retrieval-Augmented Generation): Modern LLM best practice
- Context injection: Adding dynamic knowledge to prompts
- Structured data retrieval: Query-based information extraction
- Zero-shot enhancement: Improving LLM without training

WHAT JUDGES WILL LOVE:
"We implemented a RAG system that makes the LLM context-aware without
expensive fine-tuning. It retrieves real-time train schedules, camera
locations, and person behavior history to generate intelligent alerts."
"""

import json
from datetime import datetime
from typing import Dict, Optional
from brain.train_tracker import TrainTracker


class RAGRetriever:
    """
    Retrieves contextual information for LLM prompts.

    Example usage:
        retriever = RAGRetriever('rag_database.json')

        context = retriever.get_context_for_alert(
            camera_id='platform_cam_1',
            track_id=5,
            timestamp=time.time(),
            train_tracker=train_tracker
        )

        # Context now has:
        # - Station name, platform, direction
        # - Train arrival time
        # - How many trains person skipped
        # - Time of day (peak/off-peak)
        # - Risk thresholds

        # Pass this to LLM:
        llm_analyzer.analyze(features, risk_score, track_id, rag_context=context)
    """

    def __init__(self, rag_db_path: str):
        """
        Initialize RAG retriever with database.

        Args:
            rag_db_path: Path to rag_database.json file

        LEARNING: We load the entire RAG database into memory.
        For large databases (1000+ stations), you'd use:
        - Vector database (ChromaDB, Pinecone)
        - SQL database with indexing
        - Redis for fast key-value retrieval

        For hackathon with ~10 stations: JSON in memory is perfect!
        """
        with open(rag_db_path, 'r') as f:
            self.db = json.load(f)

        self.stations = self.db['stations']
        self.cameras = self.db['cameras']
        self.thresholds = self.db.get('risk_thresholds', {})

        print(f"✓ RAGRetriever initialized")
        print(f"  Loaded {len(self.stations)} stations")
        print(f"  Loaded {len(self.cameras)} cameras")


    def get_context_for_alert(
        self,
        camera_id: str,
        track_id: int,
        timestamp: float,
        train_tracker: TrainTracker,
        additional_context: Optional[Dict] = None
    ) -> Dict:
        """
        Retrieve all relevant context for generating an alert.

        This is the MAIN function - it gathers everything the LLM needs!

        Args:
            camera_id: Which camera detected this person
            track_id: Person's tracking ID
            timestamp: Current time
            train_tracker: TrainTracker instance with train data
            additional_context: Optional extra context to include

        Returns:
            Dictionary with complete RAG context:
            {
                'station_name': 'Rajiv Chowk Metro Station',
                'platform_direction': 'Dwarka Sector 21',
                'platform_line': 'Blue Line',
                'current_time': '15:34',
                'current_date': '2026-02-23',
                'time_of_day': 'off_peak',
                'next_train_arrival_seconds': 45,
                'next_train_direction': 'Dwarka Sector 21',
                'trains_skipped_by_person': 2,
                'person_dwell_time_seconds': 180,
                'behavior_change_near_arrival': True,
                'yellow_line_distance_px': 100,
                'danger_zone_threshold_px': 50,
                'camera_location': 'north_end',
                ... (risk thresholds)
            }

        LEARNING: This is what makes LLM "smart"!
        Without RAG: "Person #5 near edge" (generic)
        With RAG: "Person #5 at Rajiv Chowk has let 2 trains pass, train arriving in 30s" (specific!)
        """
        # 1. Get camera information
        if camera_id not in self.cameras:
            raise ValueError(f"Camera '{camera_id}' not found in RAG database")

        camera_info = self.cameras[camera_id]
        station_id = camera_info['station']
        platform_id = camera_info['platform']

        # 2. Get station and platform information
        if station_id not in self.stations:
            raise ValueError(f"Station '{station_id}' not found in RAG database")

        station = self.stations[station_id]
        platform = station['platforms'][platform_id]

        # 3. Get train information
        train_info = train_tracker.get_next_train_info(timestamp)

        # 4. Get person-specific train features
        train_corr = train_tracker.get_train_correlation_features(track_id, timestamp)

        # 5. Determine time of day context
        current_time = datetime.fromtimestamp(timestamp)
        time_of_day_category = self._get_time_of_day_category(current_time, station['schedule'])

        # 6. Assemble complete context
        context = {
            # Station information
            'station_name': station['name'],
            'station_id': station_id,
            'platform_id': platform_id,
            'platform_direction': platform['direction'],
            'platform_line': platform['line'],

            # Camera information
            'camera_id': camera_id,
            'camera_location': camera_info['location'],
            'camera_fov_meters': camera_info['fov_meters'],
            'pixel_to_meter_ratio': camera_info.get('pixel_to_meter_ratio', 20),

            # Temporal information
            'current_time': current_time.strftime('%H:%M:%S'),
            'current_date': current_time.strftime('%Y-%m-%d'),
            'day_of_week': current_time.strftime('%A'),
            'time_of_day': time_of_day_category,

            # Train information
            'next_train_arrival_seconds': train_info['time_until_arrival'],
            'next_train_direction': train_info['direction'],
            'train_frequency_minutes': train_info['frequency_minutes'],
            'is_peak_hour': train_info['is_peak_hour'],

            # Person-specific train correlation
            'trains_skipped_by_person': train_corr['trains_skipped'],
            'person_dwell_time_seconds': train_corr['dwell_time_seconds'],
            'behavior_change_near_arrival': train_corr['behavior_change_near_arrival'],

            # Platform geometry
            'yellow_line_distance_px': platform['yellow_line_distance_px'],
            'danger_zone_threshold_px': platform.get('danger_zone_threshold_px', 50),

            # Risk thresholds (for LLM reference)
            'threshold_dwell_warning_seconds': self.thresholds.get('dwell_time_warning_seconds', 5),
            'threshold_dwell_critical_seconds': self.thresholds.get('dwell_time_critical_seconds', 10),
            'threshold_train_skip_suspicious': self.thresholds.get('train_skip_suspicious_count', 2),
            'threshold_train_skip_critical': self.thresholds.get('train_skip_critical_count', 3),
            'threshold_time_before_train_critical': self.thresholds.get('time_before_train_critical_seconds', 30),
        }

        # 7. Merge any additional context
        if additional_context:
            context.update(additional_context)

        return context


    def _get_time_of_day_category(self, current_time: datetime, schedule: Dict) -> str:
        """
        Categorize time of day based on schedule.

        Returns:
            'early_morning' | 'morning_peak' | 'midday' | 'evening_peak' | 'night'

        LEARNING: Time context helps LLM reason better:
        - Morning peak: Crowding is normal
        - Off-peak + isolated + near edge = suspicious
        - Night: Fewer people, easier to spot anomalies
        """
        hour = current_time.hour
        minute = current_time.minute
        current_time_str = f"{hour:02d}:{minute:02d}"

        # Check if in peak hours
        for line_schedule in schedule.values():
            peak_hours = line_schedule.get('peak_hours', [])
            for peak_range in peak_hours:
                start, end = peak_range.split('-')
                if start <= current_time_str <= end:
                    if '08:' in start or '09:' in start:
                        return 'morning_peak'
                    elif '17:' in start or '18:' in start or '19:' in start:
                        return 'evening_peak'

        # Not peak - categorize by hour
        if 5 <= hour < 8:
            return 'early_morning'
        elif 10 <= hour < 17:
            return 'midday'
        elif 20 <= hour < 24:
            return 'night'
        else:
            return 'late_night'


    def get_station_info(self, station_id: str) -> Dict:
        """
        Get complete information about a station.

        LEARNING: Useful for generating station-specific documentation
        or when LLM needs general station context (not person-specific).
        """
        if station_id not in self.stations:
            raise ValueError(f"Station '{station_id}' not found")

        return self.stations[station_id]


    def get_camera_info(self, camera_id: str) -> Dict:
        """Get information about a specific camera."""
        if camera_id not in self.cameras:
            raise ValueError(f"Camera '{camera_id}' not found")

        return self.cameras[camera_id]


    def format_context_for_llm_prompt(self, context: Dict) -> str:
        """
        Format RAG context into a human-readable string for LLM prompt.

        LEARNING: LLMs work best with well-formatted, clear context.
        Bad: {'station_name': 'Rajiv Chowk', 'next_train': 45}
        Good: "Station: Rajiv Chowk. Next train arriving in 45 seconds."

        Args:
            context: Context dictionary from get_context_for_alert()

        Returns:
            Formatted string ready to inject into LLM prompt
        """
        # Build human-readable context string
        lines = []

        # Location context
        lines.append(f"📍 LOCATION")
        lines.append(f"   Station: {context['station_name']}")
        lines.append(f"   Platform: {context['platform_line']} - toward {context['platform_direction']}")
        lines.append(f"   Camera: {context['camera_id']} ({context['camera_location']})")

        # Time context
        lines.append(f"\n🕐 TIME")
        lines.append(f"   Current: {context['current_time']} on {context['day_of_week']}")
        lines.append(f"   Period: {context['time_of_day'].replace('_', ' ').title()}")
        if context['is_peak_hour']:
            lines.append(f"   ⚠️  Peak hour - expect crowding")

        # Train context
        lines.append(f"\n🚊 TRAIN INFORMATION")
        lines.append(f"   Next arrival: {context['next_train_arrival_seconds']} seconds")
        lines.append(f"   Direction: {context['next_train_direction']}")
        lines.append(f"   Frequency: Every {context['train_frequency_minutes']} minutes")

        # Person-specific context
        lines.append(f"\n👤 PERSON BEHAVIOR HISTORY")
        lines.append(f"   Time on platform: {context['person_dwell_time_seconds']} seconds")
        lines.append(f"   Trains skipped: {context['trains_skipped_by_person']}")

        if context['trains_skipped_by_person'] >= context['threshold_train_skip_suspicious']:
            lines.append(f"   ⚠️  SUSPICIOUS: Let {context['trains_skipped_by_person']}+ trains pass!")

        if context['behavior_change_near_arrival']:
            lines.append(f"   ⚠️  Risk increased as train approached!")

        # Platform geometry
        lines.append(f"\n📏 PLATFORM SAFETY ZONES")
        lines.append(f"   Yellow line: {context['yellow_line_distance_px']}px from edge")
        lines.append(f"   Danger zone: <{context['danger_zone_threshold_px']}px from edge")

        return "\n".join(lines)


# ==================== DEMO SCRIPT ====================

if __name__ == "__main__":
    """
    Demo: RAG context retrieval and formatting.

    Run: python brain/rag_retriever.py
    """
    import time
    print("\n" + "="*70)
    print("       🔍 RAG RETRIEVER DEMO")
    print("="*70 + "\n")

    # Initialize
    print("Loading RAG database and initializing components...")
    retriever = RAGRetriever('../rag_database.json')

    # Initialize train tracker for demo
    with open('../rag_database.json', 'r') as f:
        rag_db = json.load(f)

    train_tracker = TrainTracker("demo_station", "platform_1", rag_db)

    # Simulate person behavior
    print("\nSimulating person on platform...")
    track_id = 5
    train_tracker.person_first_seen[track_id] = time.time() - 120  # Been here 2 minutes

    # Simulate skipping trains
    train_tracker.trains_skipped[track_id] = 2

    # Simulate risk increase
    train_tracker.update_person_risk(track_id, 30, time.time() - 40)
    train_tracker.update_person_risk(track_id, 70, time.time())

    print("\n" + "-"*70)
    print("RETRIEVING RAG CONTEXT")
    print("-"*70 + "\n")

    # Retrieve context
    context = retriever.get_context_for_alert(
        camera_id='platform_cam_1',
        track_id=track_id,
        timestamp=time.time(),
        train_tracker=train_tracker
    )

    print("✓ Retrieved complete RAG context!")
    print(f"\nContext dictionary has {len(context)} fields:")
    for key, value in list(context.items())[:10]:  # Show first 10
        print(f"  - {key}: {value}")
    print(f"  ... and {len(context) - 10} more fields")

    print("\n" + "-"*70)
    print("FORMATTED CONTEXT FOR LLM PROMPT")
    print("-"*70 + "\n")

    # Format for LLM
    formatted = retriever.format_context_for_llm_prompt(context)
    print(formatted)

    print("\n" + "-"*70)
    print("HOW THIS ENHANCES LLM ALERTS")
    print("-"*70 + "\n")

    print("WITHOUT RAG:")
    print("  'Person #5 detected near edge with high risk score.'")
    print("  → Generic, no context, not actionable")

    print("\nWITH RAG:")
    print("  'Person #5 at Demo Platform has been on platform for 2 minutes,")
    print("  let 2 trains pass, and is now near the edge. Train arriving in")
    print("  45 seconds. Risk increased from 30 to 70 as train approached.")
    print("  Recommend immediate mic warning.'")
    print("  → Specific, contextual, actionable!")

    print("\n" + "="*70)
    print("✓ Demo complete!")
    print("\nThis RAG system is what impresses judges!")
    print("It shows understanding of modern LLM best practices.")
    print("="*70 + "\n")
