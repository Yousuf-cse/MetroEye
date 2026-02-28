"""
Data Labeling Tool - Manual Annotation for Training
====================================================

PURPOSE:
Review your collected footage and manually label suspicious vs normal behavior.
This creates the ground-truth training dataset for XGBoost.

WORKFLOW:
1. Run your app_integrated_preview.py to collect raw data → features_dump.csv
2. Run this script to review footage and label behaviors
3. Output: labeled_training_data.csv (ready for XGBoost training)

Usage:
    python data_labeling_tool.py --video sample-station.webm --csv features_dump.csv
"""

import cv2
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
import time


class BehaviorLabeler:
    """Manual labeling tool for suspicious behavior detection."""

    def __init__(self, video_path, csv_path, window_seconds=4.0):
        self.video_path = video_path
        self.csv_path = csv_path
        self.window_seconds = window_seconds

        # Load tracking data
        print(f"Loading tracking data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} tracking records")

        # Group by track_id for windowing
        self.track_ids = self.df['track_id'].unique()
        print(f"✓ Found {len(self.track_ids)} unique tracks")

        # Storage for labeled data
        self.labeled_windows = []

    def create_windows(self):
        """
        Create aggregated windows from raw tracking data.
        Each window = 4 seconds of behavior for one person.
        """
        print("\nCreating aggregated windows...")
        windows = []

        for track_id in self.track_ids:
            track_data = self.df[self.df['track_id'] == track_id].sort_values('timestamp')

            if len(track_data) < 5:  # Skip very short tracks
                continue

            # Get window boundaries
            start_time = track_data['timestamp'].min()
            end_time = track_data['timestamp'].max()

            # Create sliding windows (every 2 seconds for overlap)
            current_time = start_time
            while current_time + self.window_seconds <= end_time:
                window_end = current_time + self.window_seconds
                window_data = track_data[
                    (track_data['timestamp'] >= current_time) &
                    (track_data['timestamp'] <= window_end)
                ]

                if len(window_data) >= 5:  # Minimum frames per window
                    agg_features = self._aggregate_window(window_data)
                    agg_features['track_id'] = int(track_id)
                    agg_features['window_start'] = current_time
                    agg_features['window_end'] = window_end
                    agg_features['camera_id'] = window_data['camera_id'].iloc[0]
                    windows.append(agg_features)

                current_time += 2.0  # Slide by 2 seconds

        print(f"✓ Created {len(windows)} windows for labeling")
        return windows

    def _aggregate_window(self, window_data):
        """Aggregate features over a time window (same as FeatureAggregator)."""
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
            'num_frames': len(window_data)
        }

        # Handle NaN values
        for key in features:
            if pd.isna(features[key]):
                features[key] = 0.0

        return features

    def start_labeling_session(self):
        """
        Interactive labeling session.
        Shows video clips and asks user to label as normal/suspicious.
        """
        windows = self.create_windows()

        if len(windows) == 0:
            print("✗ No windows to label!")
            return

        print("\n" + "="*60)
        print("LABELING SESSION STARTED")
        print("="*60)
        print("\nInstructions:")
        print("  - Watch each video clip (4 seconds)")
        print("  - Press keys to label:")
        print("    [0] = Normal behavior")
        print("    [1] = Suspicious behavior")
        print("    [s] = Skip this window")
        print("    [q] = Quit and save progress")
        print("\nSuspicious behaviors to look for:")
        print("  - Running near platform edge")
        print("  - Unusual body posture (leaning, crouching)")
        print("  - Pacing back and forth")
        print("  - Dwelling near edge for long time")
        print("  - Erratic movements")
        print("="*60 + "\n")

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        labeled_count = 0

        for i, window in enumerate(windows):
            print(f"\n[Window {i+1}/{len(windows)}]")
            print(f"  Track ID: {window['track_id']}")
            print(f"  Time: {window['window_start']:.1f}s - {window['window_end']:.1f}s")
            print(f"  Duration: {self.window_seconds}s")
            print(f"  Features:")
            print(f"    - Mean speed: {window['mean_speed']:.1f} px/s")
            print(f"    - Max speed: {window['max_speed']:.1f} px/s")
            print(f"    - Min edge distance: {window['min_dist_to_edge']:.1f} px")
            print(f"    - Dwell time: {window['dwell_time']:.1f}s")

            # Play video clip for this window
            label = self._play_and_label_clip(cap, window, fps)

            if label == 'quit':
                print("\n✓ User quit. Saving progress...")
                break
            elif label == 'skip':
                print("  → Skipped")
                continue
            else:
                window['label'] = label
                self.labeled_windows.append(window)
                labeled_count += 1
                label_text = "SUSPICIOUS" if label == 1 else "NORMAL"
                print(f"  → Labeled as: {label_text}")

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "="*60)
        print("LABELING SESSION COMPLETE")
        print("="*60)
        print(f"Total windows labeled: {labeled_count}")

        return self.labeled_windows

    def _play_and_label_clip(self, cap, window, fps):
        """Play video clip and get user label."""
        # Seek to window start time
        start_frame = int(window['window_start'] * fps)
        end_frame = int(window['window_end'] * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        print("\n  → Playing clip... (press 0=Normal, 1=Suspicious, s=Skip)")

        # Play the clip in loop until user labels
        while True:
            # Reset to start of window
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Play window
            frame_num = start_frame
            while frame_num < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Add info overlay
                cv2.putText(frame, f"Track ID: {window['track_id']}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Press 0=Normal, 1=Suspicious, s=Skip",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Labeling Tool - Press 0/1/s", frame)

                key = cv2.waitKey(30) & 0xFF

                if key == ord('0'):
                    return 0  # Normal
                elif key == ord('1'):
                    return 1  # Suspicious
                elif key == ord('s'):
                    return 'skip'
                elif key == ord('q'):
                    return 'quit'

                frame_num += 1

            # Auto-loop: replay until user labels
            print("  → Replaying clip... (press 0/1/s)")

    def save_labeled_data(self, output_path='labeled_training_data.csv'):
        """Save labeled windows to CSV for XGBoost training."""
        if len(self.labeled_windows) == 0:
            print("✗ No labeled data to save!")
            return

        df_labeled = pd.DataFrame(self.labeled_windows)

        # Keep only features needed for XGBoost
        feature_cols = [
            'track_id', 'camera_id', 'window_start', 'window_end',
            'mean_torso_angle', 'max_torso_angle', 'std_torso_angle',
            'mean_speed', 'max_speed', 'std_speed',
            'min_dist_to_edge', 'mean_dist_to_edge', 'std_dist_to_edge',
            'dwell_time', 'num_frames',
            'label'
        ]

        df_labeled = df_labeled[feature_cols]
        df_labeled.to_csv(output_path, index=False)

        print(f"\n✓ Saved {len(df_labeled)} labeled windows to: {output_path}")
        print(f"  Normal: {(df_labeled['label']==0).sum()}")
        print(f"  Suspicious: {(df_labeled['label']==1).sum()}")
        print(f"\nNext step: Train XGBoost with this data!")
        print(f"  python brain/xgboost_trainer.py")


def main():
    parser = argparse.ArgumentParser(description='Manual Labeling Tool for Behavior Detection')
    parser.add_argument('--video', type=str, required=True,
                       help='Video file to review')
    parser.add_argument('--csv', type=str, default='features_dump.csv',
                       help='CSV file with tracking data')
    parser.add_argument('--output', type=str, default='labeled_training_data.csv',
                       help='Output CSV for labeled data')
    parser.add_argument('--window', type=float, default=4.0,
                       help='Window size in seconds')

    args = parser.parse_args()

    # Create labeler
    labeler = BehaviorLabeler(
        video_path=args.video,
        csv_path=args.csv,
        window_seconds=args.window
    )

    # Start interactive labeling
    labeled_windows = labeler.start_labeling_session()

    # Save results
    if labeled_windows:
        labeler.save_labeled_data(args.output)
    else:
        print("\n✗ No data labeled. Exiting without saving.")


if __name__ == "__main__":
    main()
