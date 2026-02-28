"""
Semi-Automated Labeling Tool
=============================

PURPOSE:
Use the existing rule-based scorer to generate INITIAL labels,
then manually review and correct them. This is faster than pure manual labeling.

WORKFLOW:
1. Aggregate features from features_dump.csv
2. Use rule-based scorer to generate initial labels
3. Review and correct labels (optional)
4. Output: labeled_training_data.csv

Usage:
    # Auto-label using rules
    python auto_label_with_rules.py --csv features_dump.csv --output labeled_training_data.csv

    # Auto-label with manual review
    python auto_label_with_rules.py --csv features_dump.csv --output labeled_training_data.csv --review
"""

import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
import sys
import os

# Import the rule-based scorer
sys.path.append('brain')
from brain.rule_based_scorer import RuleBasedScorer


class AutoLabeler:
    """Automatically label behavior windows using rule-based heuristics."""

    def __init__(self, csv_path, window_seconds=4.0):
        self.csv_path = csv_path
        self.window_seconds = window_seconds
        self.scorer = RuleBasedScorer()

        # Load tracking data
        print(f"Loading tracking data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} tracking records")

        self.track_ids = self.df['track_id'].unique()
        print(f"✓ Found {len(self.track_ids)} unique tracks")

    def create_and_label_windows(self):
        """
        Create windows and automatically label them using rule-based scorer.

        Labels are based on risk level:
        - normal, low → label = 0 (normal)
        - medium, high, critical → label = 1 (suspicious)
        """
        print("\nCreating windows and auto-labeling...")
        labeled_windows = []

        for track_id in self.track_ids:
            track_data = self.df[self.df['track_id'] == track_id].sort_values('timestamp')

            if len(track_data) < 5:
                continue

            start_time = track_data['timestamp'].min()
            end_time = track_data['timestamp'].max()

            # Create sliding windows
            current_time = start_time
            while current_time + self.window_seconds <= end_time:
                window_end = current_time + self.window_seconds
                window_data = track_data[
                    (track_data['timestamp'] >= current_time) &
                    (track_data['timestamp'] <= window_end)
                ]

                if len(window_data) >= 5:
                    # Aggregate features
                    agg_features = self._aggregate_window(window_data)

                    # Use rule-based scorer to generate label
                    risk_score, risk_factors = self.scorer.compute_risk(agg_features)
                    risk_level = self.scorer.get_risk_level(risk_score)

                    # Convert risk level to binary label
                    # 0 = normal/low (safe)
                    # 1 = medium/high/critical (suspicious)
                    label = 1 if risk_level in ['medium', 'high', 'critical'] else 0

                    # Package labeled window
                    window = {
                        'track_id': int(track_id),
                        'camera_id': window_data['camera_id'].iloc[0],
                        'window_start': current_time,
                        'window_end': window_end,
                        **agg_features,
                        'risk_score': risk_score,
                        'risk_level': risk_level,
                        'label': label
                    }

                    labeled_windows.append(window)

                current_time += 2.0  # Slide by 2 seconds

        print(f"✓ Created and labeled {len(labeled_windows)} windows")

        df_labeled = pd.DataFrame(labeled_windows)
        print(f"\nLabel distribution:")
        print(f"  Normal (0): {(df_labeled['label']==0).sum()} ({(df_labeled['label']==0).sum()/len(df_labeled)*100:.1f}%)")
        print(f"  Suspicious (1): {(df_labeled['label']==1).sum()} ({(df_labeled['label']==1).sum()/len(df_labeled)*100:.1f}%)")

        return df_labeled

    def _aggregate_window(self, window_data):
        """Aggregate features over a time window."""
        features = {
            # Torso angle stats
            'mean_torso_angle': window_data['torso_angle'].mean(),
            'max_torso_angle': window_data['torso_angle'].max(),
            'min_torso_angle': window_data['torso_angle'].min(),
            'std_torso_angle': window_data['torso_angle'].std(),

            # Speed stats
            'mean_speed': window_data['speed_px_s'].mean(),
            'max_speed': window_data['speed_px_s'].max(),
            'std_speed': window_data['speed_px_s'].std(),

            # Edge distance stats
            'min_dist_to_edge': window_data['dist_to_edge_px'].min(),
            'mean_dist_to_edge': window_data['dist_to_edge_px'].mean(),
            'std_dist_to_edge': window_data['dist_to_edge_px'].std(),

            # Temporal features
            'dwell_time': window_data['dwell_time_s'].max(),
            'dwell_time_near_edge': 0.0,  # Placeholder (computed in scorer)
            'direction_changes': 0,  # Placeholder (would need position tracking)
            'num_frames': len(window_data)
        }

        # Handle NaN values
        for key in features:
            if pd.isna(features[key]):
                features[key] = 0.0

        return features

    def manual_review(self, df_labeled):
        """
        Simple console-based review of auto-labeled data.
        Shows each window and allows correction.
        """
        print("\n" + "="*60)
        print("MANUAL REVIEW MODE")
        print("="*60)
        print("\nInstructions:")
        print("  - Review each window's features and auto-assigned label")
        print("  - Press:")
        print("    [Enter] = Accept auto-label")
        print("    [0] = Change to Normal")
        print("    [1] = Change to Suspicious")
        print("    [q] = Quit review")
        print("="*60 + "\n")

        corrected_count = 0

        for idx, row in df_labeled.iterrows():
            print(f"\n[Window {idx+1}/{len(df_labeled)}]")
            print(f"  Track ID: {row['track_id']}")
            print(f"  Time: {row['window_start']:.1f}s - {row['window_end']:.1f}s")
            print(f"  Features:")
            print(f"    - Mean speed: {row['mean_speed']:.1f} px/s")
            print(f"    - Max speed: {row['max_speed']:.1f} px/s")
            print(f"    - Min edge distance: {row['min_dist_to_edge']:.1f} px")
            print(f"    - Dwell time: {row['dwell_time']:.1f}s")
            print(f"  Auto-label: {row['risk_level'].upper()} (score={row['risk_score']})")

            current_label = "SUSPICIOUS" if row['label'] == 1 else "NORMAL"
            print(f"  Assigned label: {current_label}")

            user_input = input("\n  Accept? [Enter]/0/1/q: ").strip().lower()

            if user_input == 'q':
                print("\n✓ Review stopped by user")
                break
            elif user_input == '0':
                df_labeled.at[idx, 'label'] = 0
                print("  → Changed to NORMAL")
                corrected_count += 1
            elif user_input == '1':
                df_labeled.at[idx, 'label'] = 1
                print("  → Changed to SUSPICIOUS")
                corrected_count += 1
            else:
                print("  → Accepted")

        print(f"\n✓ Review complete. Corrected {corrected_count} labels.")
        return df_labeled


def main():
    parser = argparse.ArgumentParser(description='Auto-Label Behavior Data using Rules')
    parser.add_argument('--csv', type=str, default='features_dump.csv',
                       help='CSV file with tracking data')
    parser.add_argument('--output', type=str, default='labeled_training_data.csv',
                       help='Output CSV for labeled data')
    parser.add_argument('--window', type=float, default=4.0,
                       help='Window size in seconds')
    parser.add_argument('--review', action='store_true',
                       help='Enable manual review of auto-labels')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.csv):
        print(f"✗ Error: Input file not found: {args.csv}")
        print("\nFirst run your tracking system to generate tracking data:")
        print("  python app_integrated_preview.py --video sample-station.webm")
        return

    # Create auto-labeler
    labeler = AutoLabeler(
        csv_path=args.csv,
        window_seconds=args.window
    )

    # Create and auto-label windows
    df_labeled = labeler.create_and_label_windows()

    if len(df_labeled) == 0:
        print("✗ No windows created. Check your input data.")
        return

    # Optional manual review
    if args.review:
        df_labeled = labeler.manual_review(df_labeled)

    # Save labeled data
    # Keep only columns needed for XGBoost
    feature_cols = [
        'track_id', 'camera_id', 'window_start', 'window_end',
        'mean_torso_angle', 'max_torso_angle', 'std_torso_angle',
        'mean_speed', 'max_speed', 'std_speed',
        'min_dist_to_edge', 'mean_dist_to_edge', 'std_dist_to_edge',
        'dwell_time', 'dwell_time_near_edge', 'direction_changes',
        'num_frames', 'label'
    ]

    df_final = df_labeled[feature_cols]
    df_final.to_csv(args.output, index=False)

    print("\n" + "="*60)
    print("LABELING COMPLETE")
    print("="*60)
    print(f"✓ Saved {len(df_final)} labeled windows to: {args.output}")
    print(f"\nLabel distribution:")
    print(f"  Normal (0): {(df_final['label']==0).sum()}")
    print(f"  Suspicious (1): {(df_final['label']==1).sum()}")
    print(f"\nNext steps:")
    print(f"  1. Review the labeled data (optional)")
    print(f"  2. Train XGBoost classifier:")
    print(f"     cd brain")
    print(f"     python train_xgboost_classifier.py --data ../{args.output}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
