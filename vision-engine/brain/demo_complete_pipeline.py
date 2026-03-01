"""
Complete Pipeline Demo: End-to-End Behavior Detection
======================================================

PURPOSE:
This demo shows how ALL brain components work together in a realistic scenario.
You'll see the complete flow from raw features → aggregation → scoring → LLM reasoning.

LEARNING CONCEPTS:
- Pipeline Architecture: How multiple modules connect
- Data Flow: How information passes through the system
- Integration Testing: Making sure components work together
- Real-world Usage: How your system will actually run

WHAT THIS DEMO DOES:
1. Simulates per-frame features (like what app.py produces)
2. Aggregates them into windows using FeatureAggregator
3. Scores risk using RuleBasedScorer
4. Generates natural language alerts using LLMAnalyzer
5. Shows what XGBoost training would look like (with synthetic data)

RUN THIS:
    python brain/demo_complete_pipeline.py

NOTE: This requires Ollama running! If you don't have it:
    - Demo will work with fallback messages (no LLM needed)
    - Install Ollama later: https://ollama.ai
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict

# Import our brain modules
from feature_aggregator import FeatureAggregator
from rule_based_scorer import RuleBasedScorer
from llm_analyzer import LLMAnalyzer
from xgboost_trainer import XGBoostTrainer


class PipelineDemo:
    """
    Simulates the complete behavior detection pipeline.

    LEARNING: This is what happens in your real system:
    - app.py extracts per-frame features
    - FeatureAggregator combines them into windows
    - RuleBasedScorer calculates risk
    - LLMAnalyzer generates human-readable alerts
    - XGBoostTrainer (when data ready) improves accuracy
    """

    def __init__(self):
        """Initialize all brain components."""
        print("=== Initializing Behavior Detection Pipeline ===\n")

        self.aggregator = FeatureAggregator(window_seconds=4.0)
        self.scorer = RuleBasedScorer()
        self.llm_analyzer = LLMAnalyzer()

        print("✓ Pipeline ready!\n")


    def simulate_person_behavior(
        self,
        track_id: int,
        behavior_type: str,
        num_frames: int = 120  # 4 seconds @ 30fps
    ) -> List[Dict]:
        """
        Simulate per-frame features for a person over time.

        LEARNING: In real system, app.py produces these features every frame.
        Here we simulate 3 scenarios:
        - 'normal': Person waiting safely away from edge
        - 'suspicious': Person pacing near edge, leaning
        - 'critical': Person rushing toward edge

        Args:
            track_id: Person's tracking ID
            behavior_type: 'normal', 'suspicious', or 'critical'
            num_frames: How many frames to simulate

        Returns:
            List of per-frame feature dictionaries
        """
        print(f"Simulating {behavior_type} behavior for Track #{track_id}...")

        frames = []
        start_time = time.time()

        for frame_idx in range(num_frames):
            timestamp = start_time + (frame_idx / 30.0)  # 30fps

            # Generate features based on behavior type
            if behavior_type == 'normal':
                # Normal waiting: far from edge, upright, slow movement
                bbox_center = (640 + np.random.normal(0, 5), 360 + np.random.normal(0, 5))
                torso_angle = np.random.normal(88, 2)  # Nearly vertical
                speed = np.random.normal(20, 10)  # Very slow (just fidgeting)
                dist_to_edge = np.random.normal(200, 20)  # Far from edge

            elif behavior_type == 'suspicious':
                # Suspicious: pacing near edge, moderate lean
                # Simulate back-and-forth movement
                pacing_offset = 50 * np.sin(frame_idx / 15)  # Oscillate position
                bbox_center = (640 + pacing_offset, 360 + np.random.normal(0, 3))
                torso_angle = np.random.normal(80, 5)  # Slight lean
                speed = np.random.normal(150, 40)  # Fast pacing
                dist_to_edge = np.random.normal(70, 15)  # Close to edge

            else:  # 'critical'
                # Critical: rushing toward edge, heavy lean
                # Simulate acceleration toward edge
                progress = frame_idx / num_frames
                bbox_center = (640 + progress * 100, 360 + np.random.normal(0, 10))
                torso_angle = np.random.normal(70, 3)  # Leaning forward
                speed = np.random.normal(300, 50)  # Very fast
                dist_to_edge = max(30, 150 * (1 - progress))  # Moving toward edge

            # Package into frame features (same format as app.py)
            frame_features = {
                'bbox_center': bbox_center,
                'torso_angle': torso_angle,
                'speed': speed,
                'dist_to_edge': dist_to_edge
            }

            frames.append({
                'timestamp': timestamp,
                'track_id': track_id,
                'features': frame_features
            })

        print(f"  Generated {len(frames)} frames over {num_frames/30:.1f} seconds\n")
        return frames


    def process_frames(self, frames: List[Dict], track_id: int) -> Dict:
        """
        Process per-frame features through the pipeline.

        LEARNING: This is the core logic loop!
        1. Add each frame to aggregator
        2. When window is full, get aggregated features
        3. Score the aggregated features
        4. Generate alert if risky

        Args:
            frames: List of per-frame feature dicts
            track_id: Person's tracking ID

        Returns:
            Final alert dictionary (or None if not risky)
        """
        print(f"Processing frames for Track #{track_id}...")

        alert = None

        for frame_data in frames:
            # Step 1: Add frame to aggregator
            self.aggregator.add_frame_features(
                track_id=frame_data['track_id'],
                timestamp=frame_data['timestamp'],
                features=frame_data['features']
            )

        # Step 2: Get aggregated features (after window fills)
        agg_features = self.aggregator.get_aggregated_features(track_id)

        if agg_features is None:
            print("  Window not full yet, no aggregated features.\n")
            return None

        print("  ✓ Aggregated features computed")
        print(f"    - mean_torso_angle: {agg_features['mean_torso_angle']:.1f}°")
        print(f"    - max_speed: {agg_features['max_speed']:.1f} px/s")
        print(f"    - min_dist_to_edge: {agg_features['min_dist_to_edge']:.1f} px")
        print(f"    - dwell_time_near_edge: {agg_features['dwell_time_near_edge']:.1f}s")
        print(f"    - direction_changes: {agg_features['direction_changes']}\n")

        # Step 3: Calculate risk score
        risk_score, confidence = self.scorer.compute_risk(agg_features)
        risk_level = self.scorer.get_risk_level(risk_score)

        print(f"  ✓ Risk scoring complete")
        print(f"    - Risk Score: {risk_score}/100")
        print(f"    - Risk Level: {risk_level.upper()}")
        print(f"    - Confidence: {confidence:.2f} ({confidence*100:.0f}%)\n")

        # Step 4: Generate alert if risky
        if risk_level in ['medium', 'high', 'critical']:
            print(f"  ⚠️  ALERT TRIGGERED! Calling LLM analyzer...\n")

            llm_result = self.llm_analyzer.analyze(
                features=agg_features,
                risk_score=risk_score,
                track_id=track_id,
                camera_id="demo_camera"
            )

            alert = {
                'track_id': track_id,
                'timestamp': time.time(),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'confidence': confidence,
                'features': agg_features,
                'llm_analysis': llm_result
            }

            print("  ✓ LLM Analysis complete\n")
        else:
            print(f"  ✓ No alert needed (risk level: {risk_level})\n")

        return alert


    def display_alert(self, alert: Dict):
        """
        Display alert in a human-readable format.

        LEARNING: This is what security staff would see on their dashboard.
        """
        if alert is None:
            print("No alert generated.\n")
            return

        print("\n" + "="*70)
        print("                    🚨 BEHAVIOR ALERT 🚨")
        print("="*70)
        print(f"Track ID:     #{alert['track_id']}")
        print(f"Timestamp:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert['timestamp']))}")
        print(f"Risk Level:   {alert['risk_level'].upper()}")
        print(f"Risk Score:   {alert['risk_score']}/100")
        print(f"Confidence:   {alert['confidence']:.2f} ({alert['confidence']*100:.0f}%)")
        print("-"*70)

        llm = alert['llm_analysis']
        print(f"\n💡 REASONING:")
        print(f"   {llm['reasoning']}\n")
        print(f"📢 ALERT MESSAGE:")
        print(f"   {llm['alert_message']}\n")
        print(f"🎯 RECOMMENDED ACTION:")
        print(f"   {llm['recommended_action'].replace('_', ' ').title()}\n")

        print("="*70 + "\n")


def demo_xgboost_training():
    """
    Demonstrate XGBoost training workflow (when you have labeled data).

    LEARNING: This shows what happens AFTER you collect 200+ labeled samples.
    For now, we use synthetic data. Later, you'll use real labeled CSV.
    """
    print("\n" + "="*70)
    print("           📊 BONUS: XGBoost Training Demo")
    print("="*70)
    print("\nThis shows what happens when you have labeled training data.\n")
    print("For hackathon demo, collect 200+ samples by:")
    print("  1. Running app.py on videos")
    print("  2. Labeling windows as 0 (normal) or 1 (suspicious)")
    print("  3. Saving to labeled_features.csv")
    print("  4. Running this training script\n")

    # Generate synthetic training data (normally you'd load from CSV)
    print("Generating synthetic training data for demo...\n")

    np.random.seed(42)

    # Normal behaviors
    normal_samples = pd.DataFrame({
        'mean_torso_angle': np.random.normal(88, 3, 100),
        'max_torso_angle': np.random.normal(92, 4, 100),
        'std_torso_angle': np.random.uniform(1, 4, 100),
        'mean_speed': np.random.normal(80, 20, 100),
        'max_speed': np.random.normal(120, 30, 100),
        'min_dist_to_edge': np.random.normal(180, 40, 100),
        'dwell_time_near_edge': np.random.uniform(0, 2, 100),
        'direction_changes': np.random.poisson(1.5, 100),
        'label': 0
    })

    # Suspicious behaviors
    suspicious_samples = pd.DataFrame({
        'mean_torso_angle': np.random.normal(72, 5, 50),
        'max_torso_angle': np.random.normal(68, 6, 50),
        'std_torso_angle': np.random.uniform(3, 8, 50),
        'mean_speed': np.random.normal(200, 50, 50),
        'max_speed': np.random.normal(350, 60, 50),
        'min_dist_to_edge': np.random.normal(60, 25, 50),
        'dwell_time_near_edge': np.random.uniform(5, 12, 50),
        'direction_changes': np.random.poisson(7, 50),
        'label': 1
    })

    # Combine and shuffle
    df = pd.concat([normal_samples, suspicious_samples], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Train model
    trainer = XGBoostTrainer()
    model, metrics = trainer.train(df, test_size=0.3)

    # Test prediction
    print("\n\n=== Testing Trained Model ===\n")

    test_features = {
        'mean_torso_angle': 70,
        'max_torso_angle': 65,
        'std_torso_angle': 6.0,
        'mean_speed': 220,
        'max_speed': 380,
        'min_dist_to_edge': 45,
        'dwell_time_near_edge': 9.5,
        'direction_changes': 8
    }

    pred, conf = trainer.predict(model, test_features)
    print(f"Test Prediction:")
    print(f"  Prediction: {'Suspicious' if pred == 1 else 'Normal'}")
    print(f"  Confidence: {conf:.2f} ({conf*100:.0f}%)")

    print("\n" + "="*70 + "\n")


# MAIN DEMO SCRIPT
if __name__ == "__main__":
    """
    Run complete pipeline demo with 3 scenarios.

    Command: python brain/demo_complete_pipeline.py
    """
    print("\n")
    print("="*70)
    print("     🧠 COMPLETE BRAIN PIPELINE DEMO")
    print("="*70)
    print("\nThis demo shows your entire behavior detection system in action!")
    print("We'll simulate 3 different behaviors and process them through the pipeline.\n")

    # Initialize pipeline
    demo = PipelineDemo()

    # ===== SCENARIO 1: Normal Behavior =====
    print("\n" + "="*70)
    print("SCENARIO 1: Normal Waiting Behavior")
    print("="*70)
    print("Person standing far from edge, upright posture, minimal movement.\n")

    normal_frames = demo.simulate_person_behavior(
        track_id=1,
        behavior_type='normal',
        num_frames=120
    )

    normal_alert = demo.process_frames(normal_frames, track_id=1)
    demo.display_alert(normal_alert)

    input("Press Enter to continue to Scenario 2...")


    # ===== SCENARIO 2: Suspicious Behavior =====
    print("\n" + "="*70)
    print("SCENARIO 2: Suspicious Pacing Behavior")
    print("="*70)
    print("Person pacing back and forth near edge, slight lean, high speed.\n")

    suspicious_frames = demo.simulate_person_behavior(
        track_id=2,
        behavior_type='suspicious',
        num_frames=120
    )

    suspicious_alert = demo.process_frames(suspicious_frames, track_id=2)
    demo.display_alert(suspicious_alert)

    input("Press Enter to continue to Scenario 3...")


    # ===== SCENARIO 3: Critical Behavior =====
    print("\n" + "="*70)
    print("SCENARIO 3: Critical Rushing Behavior")
    print("="*70)
    print("Person rushing toward edge, heavy forward lean, very high speed.\n")

    critical_frames = demo.simulate_person_behavior(
        track_id=3,
        behavior_type='critical',
        num_frames=120
    )

    critical_alert = demo.process_frames(critical_frames, track_id=3)
    demo.display_alert(critical_alert)


    # ===== BONUS: XGBoost Training Demo =====
    print("\n")
    choice = input("Would you like to see the XGBoost training demo? (y/n): ")
    if choice.lower() == 'y':
        demo_xgboost_training()


    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("                    ✅ DEMO COMPLETE!")
    print("="*70)
    print("\nKEY TAKEAWAYS:")
    print("1. ✓ You now have a complete working behavior detection brain!")
    print("2. ✓ The pipeline processes frames → aggregates → scores → generates alerts")
    print("3. ✓ Rule-based scoring works NOW (no ML training needed)")
    print("4. ✓ LLM adds intelligent reasoning and natural language")
    print("5. ✓ XGBoost training is ready when you have labeled data\n")

    print("NEXT STEPS:")
    print("1. Integrate this brain logic with your app.py (YOLO detection)")
    print("2. Collect real data from your public filming sessions")
    print("3. Label 200+ samples and train XGBoost")
    print("4. Build FastAPI backend (Days 3-4 of roadmap)")
    print("5. Build React frontend (Days 4-5 of roadmap)")
    print("6. Complete end-to-end integration (Day 6 of roadmap)\n")

    print("For now, your brain logic is COMPLETE and ready to use! 🎉\n")
    print("="*70 + "\n")
