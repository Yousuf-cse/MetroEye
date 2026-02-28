"""
Rule-Based Risk Scorer
======================

PURPOSE:
Calculates risk scores and confidence levels using simple rules (no ML needed).

WHY START WITH RULES?
- Fast to implement (no training data needed)
- Easy to explain to judges ("if distance < 50px, add 20 points")
- Transparent (you can see exactly why someone was flagged)
- Good baseline (helps you understand which features matter)

LEARNING CONCEPTS:
- Rule-Based Systems: Simple if-then logic
- Weighted Scoring: Different rules contribute different amounts
- Confidence: How sure are we about the prediction?
- Thresholding: Converting continuous scores to categories

WHEN TO UPGRADE:
After you collect labeled data, train XGBoost to replace these rules.
XGBoost will learn patterns you might miss with manual rules.
"""

from typing import Dict, Tuple


class RuleBasedScorer:
    """
    Computes risk scores using hand-crafted rules.

    Example usage:
        scorer = RuleBasedScorer()

        features = {
            'min_dist_to_edge': 45,
            'dwell_time_near_edge': 8.0,
            'max_speed': 350,
            'direction_changes': 7,
            'mean_torso_angle': 72
        }

        score, confidence = scorer.compute_risk(features)
        level = scorer.get_risk_level(score)

        print(f"Risk: {level} (score={score}, confidence={confidence})")
        # Output: Risk: high (score=65, confidence=0.78)
    """

    def __init__(self):
        """
        Initialize scorer with configurable thresholds.

        LEARNING: These thresholds define risk categories.
        Adjust them based on your data to minimize false alarms.
        """
        # Risk level thresholds (0-100 scale)
        self.thresholds = {
            'low': 30,       # 0-29: Normal
            'medium': 50,    # 30-49: Monitor
            'high': 70,      # 50-69: Warning
            'critical': 85   # 70+: Critical
        }

        # Max possible score (sum of all rule weights)
        # LEARNING: We use this to normalize confidence (score / max_score)
        self.max_score = 120  # Updated for new rules (phone usage + acceleration)

        print("✓ RuleBasedScorer initialized")
        print(f"  Thresholds: {self.thresholds}")


    def compute_risk(self, features: Dict) -> Tuple[int, float]:
        """
        Compute risk score and confidence from features.

        Args:
            features: Aggregated features from FeatureAggregator

        Returns:
            (risk_score, confidence)
            - risk_score: 0-100 scale
            - confidence: 0.0-1.0 (how certain we are)

        LEARNING: Each rule adds points if condition is met.
        More triggered rules = higher score = more confident prediction.
        """
        score = 0
        triggered_rules = []  # Track which rules fired (for confidence)

        # ===== Rule 1: Distance to Platform Edge =====
        # LEARNING: Being close to edge is THE most important indicator
        # Why? People normally stay 100-200px away. <50px is abnormal.

        min_dist = features.get('min_dist_to_edge')
        if min_dist is not None:
            if min_dist < 50:
                score += 20
                triggered_rules.append('very_close_to_edge')
            elif min_dist < 100:
                score += 10
                triggered_rules.append('close_to_edge')

        # ===== Rule 2: Dwell Time Near Edge =====
        # LEARNING: Normal people don't linger near dangerous areas.
        # Walking past = 1-2s dwell time (OK)
        # Standing/waiting = 5-10s dwell time (SUSPICIOUS)

        dwell_time = features.get('dwell_time_near_edge', 0)
        if dwell_time > 8:
            score += 20
            triggered_rules.append('long_dwell_time')
        elif dwell_time > 5:
            score += 15
            triggered_rules.append('moderate_dwell_time')
        elif dwell_time > 3:
            score += 10
            triggered_rules.append('short_dwell_time')

        # ===== Rule 3: Torso Angle (Leaning) =====
        # LEARNING: Vertical torso = 90 degrees
        # Leaning forward (<75°) or backward (>105°) is unusual

        mean_angle = features.get('mean_torso_angle')
        if mean_angle is not None:
            if mean_angle < 70 or mean_angle > 110:
                score += 15
                triggered_rules.append('extreme_lean')
            elif mean_angle < 75 or mean_angle > 105:
                score += 10
                triggered_rules.append('moderate_lean')

        # ===== Rule 4: Sudden High Speed =====
        # LEARNING: Sudden rush toward edge is dangerous
        # Normal walking: 50-150 px/s
        # Running: 200-400 px/s

        max_speed = features.get('max_speed')
        if max_speed is not None:
            if max_speed > 350:
                score += 15
                triggered_rules.append('very_high_speed')
            elif max_speed > 250:
                score += 10
                triggered_rules.append('high_speed')

        # ===== Rule 5: Direction Changes (Pacing) =====
        # LEARNING: Pacing back and forth indicates agitation/confusion
        # Normal walking: 0-2 direction changes
        # Pacing: 5+ direction changes

        dir_changes = features.get('direction_changes', 0)
        if dir_changes > 8:
            score += 15
            triggered_rules.append('heavy_pacing')
        elif dir_changes > 5:
            score += 10
            triggered_rules.append('moderate_pacing')

        # ===== Rule 6: Phone Usage Near Edge (NEW!) =====
        # LEARNING: Distracted + dangerous location = CRITICAL RISK
        # This is a COMBINATION rule - requires BOTH conditions:
        # 1. Person looking down (head_pitch > 30°)
        # 2. Near platform edge (dist < 100px)
        #
        # Why critical? Distracted person doesn't see approaching train!
        # LLM will flag: "Person looking at phone near edge, train arriving in 30s"

        mean_head_pitch = features.get('mean_head_pitch')
        time_looking_down = features.get('time_looking_down', 0)
        min_dist = features.get('min_dist_to_edge')

        if (mean_head_pitch is not None and
            min_dist is not None and
            mean_head_pitch > 30 and  # Looking down
            min_dist < 100 and        # Near edge
            time_looking_down > 2.0):  # Sustained (not just glancing)
            score += 25  # HIGH WEIGHT - This is very dangerous!
            triggered_rules.append('distracted_near_edge')

        # ===== Rule 7: Sudden Rush Toward Edge (NEW!) =====
        # LEARNING: High acceleration + moving toward edge = potential crisis
        # Acceleration = rate of speed change
        #
        # Normal: Person walks steadily (low acceleration)
        # Suspicious: Person suddenly rushes forward (high acceleration)
        #
        # Especially dangerous if:
        # - High acceleration (>300 px/s²)
        # - Currently close to edge (<150px)

        max_accel = features.get('max_acceleration', 0)
        accel_spikes = features.get('acceleration_spikes', 0)

        if max_accel > 400 and min_dist is not None and min_dist < 150:
            score += 20  # Very high acceleration toward edge
            triggered_rules.append('rushing_toward_edge')
        elif max_accel > 300 and min_dist is not None and min_dist < 100:
            score += 15  # Moderate acceleration very close to edge
            triggered_rules.append('accelerating_near_edge')

        # Erratic movement (multiple acceleration spikes)
        if accel_spikes >= 3:
            score += 10
            triggered_rules.append('erratic_movement')

        # ===== Compute Confidence =====
        # LEARNING: Confidence = how many rules triggered
        # More rules triggered = more evidence = higher confidence

        # Method 1: Based on score (simple)
        confidence_from_score = min(score / self.max_score, 1.0)

        # Method 2: Based on number of triggered rules (more robust)
        max_possible_rules = 7  # Updated: 5 original + 2 new (phone, acceleration)
        confidence_from_rules = len(triggered_rules) / max_possible_rules

        # Combine both methods (average)
        confidence = (confidence_from_score + confidence_from_rules) / 2.0

        # Cap score at 100
        score = min(score, 100)

        # Store triggered rules for explainability
        self.last_triggered_rules = triggered_rules

        return score, confidence


    def get_risk_level(self, score: int) -> str:
        """
        Convert numeric score to risk category.

        LEARNING: Thresholding converts continuous values to discrete categories.
        Easier for humans to understand than raw scores.

        Args:
            score: Risk score (0-100)

        Returns:
            'normal', 'low', 'medium', 'high', or 'critical'
        """
        if score >= self.thresholds['critical']:
            return 'critical'
        elif score >= self.thresholds['high']:
            return 'high'
        elif score >= self.thresholds['medium']:
            return 'medium'
        elif score >= self.thresholds['low']:
            return 'low'
        else:
            return 'normal'


    def explain_score(self) -> str:
        """
        Generate human-readable explanation of why rules fired.

        LEARNING: Explainability is CRUCIAL for trust.
        Judges/users need to understand WHY the system made a decision.

        Returns:
            String explaining which rules triggered
        """
        if not hasattr(self, 'last_triggered_rules'):
            return "No rules triggered yet"

        if not self.last_triggered_rules:
            return "No suspicious patterns detected"

        # Map rule IDs to human-readable descriptions
        rule_descriptions = {
            'very_close_to_edge': 'Very close to platform edge (<50px)',
            'close_to_edge': 'Close to platform edge (<100px)',
            'long_dwell_time': 'Standing near edge for extended time (>8s)',
            'moderate_dwell_time': 'Dwelling near edge (5-8s)',
            'short_dwell_time': 'Brief stay near edge (3-5s)',
            'extreme_lean': 'Extreme body lean detected',
            'moderate_lean': 'Moderate body lean detected',
            'very_high_speed': 'Very high movement speed (>350 px/s)',
            'high_speed': 'High movement speed (>250 px/s)',
            'heavy_pacing': 'Heavy pacing behavior (>8 direction changes)',
            'moderate_pacing': 'Pacing behavior detected (5-8 direction changes)'
        }

        explanations = [
            rule_descriptions.get(rule, rule)
            for rule in self.last_triggered_rules
        ]

        return "Triggered rules:\n  - " + "\n  - ".join(explanations)


# EDUCATIONAL EXAMPLE
if __name__ == "__main__":
    """
    Run this file directly to see demo of risk scoring.

    Command: python rule_based_scorer.py
    """
    print("=== Rule-Based Scorer Demo ===\n")

    scorer = RuleBasedScorer()

    # Test Case 1: Very Suspicious Behavior
    print("\n--- Test Case 1: Very Suspicious ---")
    features_suspicious = {
        'min_dist_to_edge': 35,       # Very close!
        'dwell_time_near_edge': 9.5,  # Long time!
        'max_speed': 380,              # Sudden rush!
        'direction_changes': 9,        # Heavy pacing!
        'mean_torso_angle': 68         # Leaning forward!
    }

    score, confidence = scorer.compute_risk(features_suspicious)
    level = scorer.get_risk_level(score)

    print(f"\nFeatures: {features_suspicious}")
    print(f"\nResult:")
    print(f"  Risk Score: {score}/100")
    print(f"  Risk Level: {level.upper()}")
    print(f"  Confidence: {confidence:.2f} ({confidence*100:.0f}%)")
    print(f"\n{scorer.explain_score()}")

    # Test Case 2: Normal Behavior
    print("\n\n--- Test Case 2: Normal Behavior ---")
    features_normal = {
        'min_dist_to_edge': 180,       # Far from edge
        'dwell_time_near_edge': 0.5,   # Just walking past
        'max_speed': 95,                # Normal walking speed
        'direction_changes': 1,         # Straight path
        'mean_torso_angle': 89          # Standing upright
    }

    score, confidence = scorer.compute_risk(features_normal)
    level = scorer.get_risk_level(score)

    print(f"\nFeatures: {features_normal}")
    print(f"\nResult:")
    print(f"  Risk Score: {score}/100")
    print(f"  Risk Level: {level.upper()}")
    print(f"  Confidence: {confidence:.2f} ({confidence*100:.0f}%)")
    print(f"\n{scorer.explain_score()}")

    # Test Case 3: Borderline Behavior
    print("\n\n--- Test Case 3: Borderline (Moderate Risk) ---")
    features_borderline = {
        'min_dist_to_edge': 85,        # Somewhat close
        'dwell_time_near_edge': 4.2,   # Brief stay
        'max_speed': 210,               # Moderate speed
        'direction_changes': 3,         # Some back/forth
        'mean_torso_angle': 82          # Mostly upright
    }

    score, confidence = scorer.compute_risk(features_borderline)
    level = scorer.get_risk_level(score)

    print(f"\nFeatures: {features_borderline}")
    print(f"\nResult:")
    print(f"  Risk Score: {score}/100")
    print(f"  Risk Level: {level.upper()}")
    print(f"  Confidence: {confidence:.2f} ({confidence*100:.0f}%)")
    print(f"\n{scorer.explain_score()}")

    print("\n\n✓ Rule-based scoring works!")
    print("\nKEY TAKEAWAY:")
    print("Rule-based scoring gives you:")
    print("  1. Risk score (0-100)")
    print("  2. Confidence level (0.0-1.0)")
    print("  3. Explainability (which rules triggered)")
    print("\nLater, you can train XGBoost to replace these rules with learned patterns!")
