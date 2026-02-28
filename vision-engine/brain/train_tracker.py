"""
Train Arrival Tracker - RAG Context Enhancement
================================================

PURPOSE:
Tracks train arrivals and correlates them with person behavior patterns.
This provides crucial context for the LLM to make better risk assessments.

KEY FEATURES:
1. Predicts next train arrival based on schedule
2. Counts how many trains each person has let pass (suspicious if 2+)
3. Detects behavior changes around train arrivals
4. Supports both scheduled times and real-time API integration

LEARNING CONCEPTS:
- RAG (Retrieval-Augmented Generation): Injecting context into LLM prompts
- Temporal correlation: Risk patterns related to events (train arrivals)
- Domain knowledge integration: Using metro-specific context
- State tracking: Managing dynamic information across time

WHY THIS MATTERS:
- Person letting 2+ trains pass = suspicious pattern
- Behavior change when train arrives = high risk indicator
- Time-aware alerts: "Train arriving in 30 seconds" makes LLM smarter
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import json


class TrainTracker:
    """
    Tracks train arrivals and person-train interaction patterns.

    Example usage:
        tracker = TrainTracker("rajiv_chowk", "platform_1", rag_db)

        # During video processing
        next_train = tracker.get_next_train_info()
        print(f"Next train in {next_train['time_until_arrival']}s")

        # When train arrives (detect via computer vision or manual trigger)
        tracker.on_train_arrived([5, 7, 12])  # track_ids of people on platform

        # Get features for a specific person
        features = tracker.get_train_correlation_features(track_id=5)
        print(f"Person #5 has skipped {features['trains_skipped']} trains")
    """

    def __init__(self, station_id: str, platform_id: str, rag_db: dict):
        """
        Initialize train tracker.

        Args:
            station_id: Station identifier (e.g., "rajiv_chowk")
            platform_id: Platform identifier (e.g., "platform_1")
            rag_db: RAG database dictionary loaded from JSON

        LEARNING: We load schedule data from RAG database to predict arrivals
        """
        self.station_id = station_id
        self.platform_id = platform_id

        # Load station and schedule data
        station = rag_db['stations'][station_id]
        platform = station['platforms'][platform_id]

        self.line = platform['line']  # e.g., "Blue Line"
        self.direction = platform['direction']  # e.g., "Dwarka Sector 21"

        # Find schedule for this line
        line_key = self.line.lower().replace(' ', '_')
        self.schedule = station['schedule'].get(line_key, {})

        # Train tracking state
        self.last_train_time = None  # Last actual train arrival
        self.next_train_time = None  # Predicted next arrival
        self.train_count_today = 0

        # Track how many trains each person has skipped
        # LEARNING: defaultdict creates empty value (0) for new keys automatically
        self.trains_skipped = defaultdict(int)  # track_id -> skip count

        # Track when each person first appeared (for behavior change detection)
        self.person_first_seen = {}  # track_id -> timestamp
        self.person_risk_history = defaultdict(list)  # track_id -> list of (timestamp, risk_score)

        # Compute first train arrival
        self._compute_next_arrival(time.time())

        print(f"✓ TrainTracker initialized")
        print(f"  Station: {station_id}, Platform: {platform_id}")
        print(f"  Line: {self.line}, Direction: {self.direction}")
        print(f"  Frequency: {self.schedule.get('frequency_minutes', 'N/A')} min")


    def _compute_next_arrival(self, current_timestamp: float):
        """
        Compute next train arrival time based on schedule.

        LEARNING: Two approaches:
        1. Demo mode: Simple time-based calculation from frequency
        2. Production mode: Would call real-time train API

        For hackathon, we use approach #1 (simpler, works offline)
        """
        if not self.schedule:
            # No schedule data, assume trains every 5 minutes
            self.next_train_time = current_timestamp + (5 * 60)
            return

        frequency_minutes = self.schedule.get('frequency_minutes', 5)

        # Check if we're in peak hours
        current_time = datetime.fromtimestamp(current_timestamp)
        current_time_str = current_time.strftime('%H:%M')

        in_peak = False
        peak_hours = self.schedule.get('peak_hours', [])
        for peak_range in peak_hours:
            start, end = peak_range.split('-')
            if start <= current_time_str <= end:
                in_peak = True
                break

        if in_peak:
            frequency_minutes = self.schedule.get('peak_frequency_minutes', frequency_minutes)

        # Calculate next arrival
        if self.last_train_time is None:
            # First prediction: Next train arrives in [frequency] minutes
            self.next_train_time = current_timestamp + (frequency_minutes * 60)
        else:
            # After a train arrives, next one is [frequency] minutes later
            self.next_train_time = self.last_train_time + (frequency_minutes * 60)


    def get_next_train_info(self, current_timestamp: Optional[float] = None) -> Dict:
        """
        Get information about next train arrival.

        Args:
            current_timestamp: Current time (default: time.time())

        Returns:
            Dictionary with:
            - time_until_arrival: Seconds until next train
            - direction: Where train is going
            - frequency: How often trains come (minutes)
            - is_peak_hour: Whether we're in peak hours

        LEARNING: This is what gets injected into LLM RAG context!
        """
        if current_timestamp is None:
            current_timestamp = time.time()

        # Recompute if prediction is stale
        if self.next_train_time is None or self.next_train_time < current_timestamp:
            self._compute_next_arrival(current_timestamp)

        time_until_arrival = max(0, self.next_train_time - current_timestamp)

        # Check if peak hour
        current_time = datetime.fromtimestamp(current_timestamp)
        current_time_str = current_time.strftime('%H:%M')

        in_peak = False
        peak_hours = self.schedule.get('peak_hours', [])
        for peak_range in peak_hours:
            start, end = peak_range.split('-')
            if start <= current_time_str <= end:
                in_peak = True
                break

        return {
            'time_until_arrival': int(time_until_arrival),
            'direction': self.direction,
            'frequency_minutes': self.schedule.get('frequency_minutes', 5),
            'is_peak_hour': in_peak,
            'train_count_today': self.train_count_today
        }


    def on_train_arrived(self, track_ids_on_platform: List[int], current_timestamp: Optional[float] = None):
        """
        Called when a train arrives at the platform.

        For each person on the platform who DOESN'T board (i.e., still tracked after train leaves),
        we increment their skip counter.

        Args:
            track_ids_on_platform: List of track IDs of people currently on platform
            current_timestamp: When train arrived (default: time.time())

        LEARNING: This is typically called manually or via computer vision
        (detecting train entering frame). For demo, you can simulate it!

        Usage:
            # When you detect train arrival (manual or CV)
            people_on_platform = list(aggregator.track_windows.keys())
            tracker.on_train_arrived(people_on_platform)

            # Wait 30 seconds (train stops, boards, leaves)
            time.sleep(30)

            # People still on platform skipped the train
            people_still_here = list(aggregator.track_windows.keys())
            for track_id in people_still_here:
                if track_id in people_on_platform:
                    # This person was here before AND after train = skipped it
                    # (already incremented in on_train_arrived)
                    pass
        """
        if current_timestamp is None:
            current_timestamp = time.time()

        # Record train arrival
        self.last_train_time = current_timestamp
        self.train_count_today += 1

        # Increment skip counter for everyone on platform
        # LEARNING: Assumption is that people who stay are skipping the train
        # In production, you'd track who actually boards vs who stays
        for track_id in track_ids_on_platform:
            self.trains_skipped[track_id] += 1

        # Compute next train arrival
        self._compute_next_arrival(current_timestamp)

        print(f"🚊 Train arrived! {len(track_ids_on_platform)} people on platform")
        print(f"   Next train in {int(self.next_train_time - current_timestamp)}s")


    def person_boarded_train(self, track_id: int):
        """
        Call this when a person boards the train (leaves the platform).

        This decrements their skip counter (they didn't skip, they waited and boarded).

        LEARNING: In production CV system, you'd detect this by:
        - Person's bounding box disappears for >5 seconds
        - Person moves into train door region
        - Track lost permanently (not temporarily occluded)

        Args:
            track_id: ID of person who boarded
        """
        if track_id in self.trains_skipped and self.trains_skipped[track_id] > 0:
            self.trains_skipped[track_id] -= 1  # They waited for a train and took it

        # Clean up history
        if track_id in self.person_first_seen:
            del self.person_first_seen[track_id]
        if track_id in self.person_risk_history:
            del self.person_risk_history[track_id]


    def update_person_risk(self, track_id: int, risk_score: int, timestamp: Optional[float] = None):
        """
        Track risk score changes for a person over time.

        Used to detect behavior changes near train arrivals.

        Args:
            track_id: Person's ID
            risk_score: Current risk score (0-100)
            timestamp: When this risk was recorded

        LEARNING: By tracking risk history, we can detect patterns like:
        "Person was calm, but risk jumped to 80 when train approached"
        """
        if timestamp is None:
            timestamp = time.time()

        if track_id not in self.person_first_seen:
            self.person_first_seen[track_id] = timestamp

        self.person_risk_history[track_id].append((timestamp, risk_score))

        # Keep only last 10 risk scores (sliding window)
        if len(self.person_risk_history[track_id]) > 10:
            self.person_risk_history[track_id].pop(0)


    def get_train_correlation_features(self, track_id: int, current_timestamp: Optional[float] = None) -> Dict:
        """
        Get train-related features for a specific person.

        This is the RAG context that gets injected into the LLM prompt!

        Args:
            track_id: Person's ID
            current_timestamp: Current time

        Returns:
            Dictionary with:
            - trains_skipped: How many trains this person let pass
            - time_until_next_train_seconds: Seconds until next arrival
            - behavior_change_near_arrival: Did risk increase when train approached?
            - dwell_time_seconds: How long they've been on platform

        LEARNING: These features help LLM reason better:
        "Person #5 has let 2 trains pass and train arriving in 30s = SUSPICIOUS"
        """
        if current_timestamp is None:
            current_timestamp = time.time()

        trains_skipped = self.trains_skipped.get(track_id, 0)

        # Time until next train
        train_info = self.get_next_train_info(current_timestamp)
        time_until_next = train_info['time_until_arrival']

        # Dwell time (how long on platform)
        dwell_time = 0
        if track_id in self.person_first_seen:
            dwell_time = current_timestamp - self.person_first_seen[track_id]

        # Behavior change detection
        behavior_change = self._detect_behavior_change_near_train(track_id, current_timestamp)

        return {
            'trains_skipped': trains_skipped,
            'time_until_next_train_seconds': time_until_next,
            'behavior_change_near_arrival': behavior_change,
            'dwell_time_seconds': int(dwell_time),
            'is_peak_hour': train_info['is_peak_hour']
        }


    def _detect_behavior_change_near_train(self, track_id: int, current_timestamp: float) -> bool:
        """
        Detect if person's risk increased when train approached.

        LEARNING: This is a simple heuristic. Production version would use:
        - Statistical change detection (CUSUM, Bayesian changepoint)
        - Time-series anomaly detection
        - Correlation analysis

        Returns:
            True if risk increased significantly in last 30 seconds
        """
        if track_id not in self.person_risk_history:
            return False

        history = self.person_risk_history[track_id]
        if len(history) < 3:
            return False

        # Get recent risk scores (last 30 seconds)
        recent_scores = [(ts, score) for ts, score in history if current_timestamp - ts <= 30]

        if len(recent_scores) < 2:
            return False

        # Check if risk increased significantly
        oldest_recent = recent_scores[0][1]
        newest_recent = recent_scores[-1][1]

        # Risk jumped by 20+ points = behavior change
        if newest_recent - oldest_recent >= 20:
            # Check if train is approaching (within 60 seconds)
            train_info = self.get_next_train_info(current_timestamp)
            if train_info['time_until_arrival'] <= 60:
                return True

        return False


    def reset_person(self, track_id: int):
        """
        Reset all tracking data for a person (when they leave platform).

        LEARNING: Call this when:
        - Track is lost for >5 seconds
        - Person boards train
        - Person leaves camera FOV
        """
        if track_id in self.trains_skipped:
            del self.trains_skipped[track_id]
        if track_id in self.person_first_seen:
            del self.person_first_seen[track_id]
        if track_id in self.person_risk_history:
            del self.person_risk_history[track_id]


    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """
        Remove tracking data for people no longer on platform.

        LEARNING: Prevents memory leaks in long-running systems!

        Args:
            active_track_ids: List of currently tracked person IDs
        """
        all_tracked = set(list(self.trains_skipped.keys()) +
                         list(self.person_first_seen.keys()) +
                         list(self.person_risk_history.keys()))

        for track_id in all_tracked:
            if track_id not in active_track_ids:
                self.reset_person(track_id)


# ==================== DEMO SCRIPT ====================

if __name__ == "__main__":
    """
    Demo: Train tracking and skip counting.

    Run: python brain/train_tracker.py
    """
    print("\n" + "="*70)
    print("       🚊 TRAIN TRACKER DEMO")
    print("="*70 + "\n")

    # Load RAG database
    print("Loading RAG database...")
    with open('../rag_database.json', 'r') as f:
        rag_db = json.load(f)

    # Initialize tracker
    tracker = TrainTracker("demo_station", "platform_1", rag_db)

    print("\n" + "-"*70)
    print("SCENARIO: Person #5 arrives on platform")
    print("-"*70 + "\n")

    # Simulate person arriving on platform
    tracker.person_first_seen[5] = time.time()

    # Check next train
    train_info = tracker.get_next_train_info()
    print(f"Next train info:")
    print(f"  - Arriving in: {train_info['time_until_arrival']}s")
    print(f"  - Direction: {train_info['direction']}")
    print(f"  - Frequency: {train_info['frequency_minutes']} minutes")
    print(f"  - Peak hour: {train_info['is_peak_hour']}")

    # Get initial features
    features = tracker.get_train_correlation_features(5)
    print(f"\nPerson #5 features:")
    print(f"  - Trains skipped: {features['trains_skipped']}")
    print(f"  - Time on platform: {features['dwell_time_seconds']}s")

    input("\nPress Enter to simulate first train arrival...")

    print("\n" + "-"*70)
    print("SCENARIO: First train arrives, Person #5 doesn't board")
    print("-"*70 + "\n")

    # Train arrives
    tracker.on_train_arrived([5])  # Person 5 is on platform

    # Check updated features
    features = tracker.get_train_correlation_features(5)
    print(f"Person #5 after Train #1:")
    print(f"  - Trains skipped: {features['trains_skipped']} ← INCREASED!")
    print(f"  - Next train in: {features['time_until_next_train_seconds']}s")

    input("\nPress Enter to simulate second train arrival...")

    print("\n" + "-"*70)
    print("SCENARIO: Second train arrives, Person #5 STILL doesn't board")
    print("-"*70 + "\n")

    # Simulate time passing
    time.sleep(1)

    # Second train arrives
    tracker.on_train_arrived([5])

    # Check updated features
    features = tracker.get_train_correlation_features(5)
    print(f"Person #5 after Train #2:")
    print(f"  - Trains skipped: {features['trains_skipped']} ← SUSPICIOUS!")
    print(f"  - Next train in: {features['time_until_next_train_seconds']}s")
    print(f"  - Time on platform: {features['dwell_time_seconds']}s")

    print("\n⚠️  ALERT: Person has let 2+ trains pass!")
    print("   LLM will use this context to assess higher risk.")

    input("\nPress Enter to simulate risk increase...")

    print("\n" + "-"*70)
    print("SCENARIO: Person #5's risk increases as train approaches")
    print("-"*70 + "\n")

    # Simulate risk increase
    tracker.update_person_risk(5, 30, time.time() - 35)  # Low risk 35s ago
    tracker.update_person_risk(5, 35, time.time() - 20)  # Still low 20s ago
    tracker.update_person_risk(5, 65, time.time())       # HIGH risk now!

    features = tracker.get_train_correlation_features(5)
    print(f"Person #5 behavior change:")
    print(f"  - Behavior change near arrival: {features['behavior_change_near_arrival']}")
    print(f"  - Risk jumped from 30 → 65 as train approached!")

    print("\n🚨 CRITICAL: Person skipped 2 trains + behavior changed + train arriving soon!")
    print("   LLM will generate high-priority alert with this RAG context.")

    print("\n" + "="*70)
    print("✓ Demo complete! This RAG context makes LLM alerts much smarter.")
    print("="*70 + "\n")
