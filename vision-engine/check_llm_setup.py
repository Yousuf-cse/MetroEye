#!/usr/bin/env python3
"""
LLM Setup Checker
=================

This script diagnoses why LLM analysis is not working in the vision engine.
Checks:
1. Brain modules can be imported
2. Ollama is running
3. Model is available
4. LLM analyzer can generate analysis
5. Features are being extracted properly
"""

import sys
import requests
import time

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_brain_imports():
    """Check if brain modules can be imported"""
    print_header("1. Checking Brain Module Imports")

    try:
        from brain.feature_aggregator import FeatureAggregator
        print("✅ FeatureAggregator imported successfully")
    except ImportError as e:
        print(f"❌ Cannot import FeatureAggregator: {e}")
        return False

    try:
        from brain.rule_based_scorer import RuleBasedScorer
        print("✅ RuleBasedScorer imported successfully")
    except ImportError as e:
        print(f"❌ Cannot import RuleBasedScorer: {e}")
        return False

    try:
        from brain.llm_analyzer import LLMAnalyzer
        print("✅ LLMAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ Cannot import LLMAnalyzer: {e}")
        return False

    print("\n✅ All brain modules can be imported!")
    return True

def check_ollama_running():
    """Check if Ollama server is running"""
    print_header("2. Checking Ollama Server")

    try:
        response = requests.get("http://localhost:11434/api/version", timeout=2.0)
        if response.ok:
            version_data = response.json()
            print(f"✅ Ollama is running!")
            print(f"   Version: {version_data.get('version', 'unknown')}")
            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama on http://localhost:11434")
        print("\n💡 To fix:")
        print("   1. Install Ollama: https://ollama.com/download")
        print("   2. Start server: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def check_model_available():
    """Check if the phi3:mini model is available"""
    print_header("3. Checking Model Availability")

    try:
        # Try to generate with the model
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": "Say hello in one word",
                "stream": False,
                "options": {
                    "num_predict": 5
                }
            },
            timeout=10.0
        )

        if response.status_code == 404:
            print("❌ Model 'phi3:mini' not found")
            print("\n💡 To fix:")
            print("   ollama pull phi3:mini")
            return False
        elif response.ok:
            result = response.json()
            generated = result.get('response', '').strip()
            print(f"✅ Model 'phi3:mini' is available!")
            print(f"   Test response: '{generated}'")
            return True
        else:
            print(f"❌ Model check failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error checking model: {e}")
        return False

def test_llm_analyzer():
    """Test if LLMAnalyzer can generate alerts"""
    print_header("4. Testing LLM Analyzer")

    try:
        from brain.llm_analyzer import LLMAnalyzer

        # Initialize analyzer
        print("Initializing LLMAnalyzer...")
        analyzer = LLMAnalyzer()

        # Test features
        test_features = {
            'window_duration': 4.5,
            'mean_torso_angle': 72.5,
            'mean_speed': 180.0,
            'max_speed': 385.0,
            'min_dist_to_edge': 42.0,
            'dwell_time_near_edge': 8.5,
            'direction_changes': 9,
            'max_acceleration': 320.0,
            'acceleration_spikes': 4
        }

        risk_score = 75
        track_id = 999

        print(f"\nTest Input:")
        print(f"  Track ID: {track_id}")
        print(f"  Risk Score: {risk_score}")
        print(f"  Features: {list(test_features.keys())}")

        print("\n⏳ Calling LLM... (this may take 3-10 seconds)")
        start_time = time.time()

        result = analyzer.analyze(
            features=test_features,
            risk_score=risk_score,
            track_id=track_id,
            camera_id="test_camera"
        )

        elapsed = time.time() - start_time

        print(f"\n✅ LLM Analysis successful! (took {elapsed:.2f}s)")
        print(f"\nResult:")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Alert Message: {result['alert_message']}")
        print(f"  Recommended Action: {result['recommended_action']}")
        print(f"\n  Reasoning:")
        print(f"    {result['reasoning']}")

        return True

    except Exception as e:
        print(f"❌ LLM Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_aggregation():
    """Test if feature aggregation works"""
    print_header("5. Testing Feature Aggregation")

    try:
        from brain.feature_aggregator import FeatureAggregator

        aggregator = FeatureAggregator(window_seconds=4.0)
        track_id = 123

        # Simulate adding features over time
        print("Simulating 5 frames of tracking data...")
        for i in range(5):
            frame_features = {
                'bbox_center': (500 + i*10, 300 + i*5),
                'torso_angle': 85.0 - i*2,
                'speed': 150.0 + i*20,
                'dist_to_edge': 80.0 - i*5
            }
            aggregator.add_frame_features(track_id, time.time() + i*0.1, frame_features)

        # Get aggregated features
        agg_features = aggregator.get_aggregated_features(track_id)

        if agg_features:
            print(f"\n✅ Feature aggregation works!")
            print(f"\nAggregated Features:")
            for key, value in agg_features.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            return True
        else:
            print("❌ No aggregated features returned")
            return False

    except Exception as e:
        print(f"❌ Feature aggregation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*70)
    print("  METROEYE LLM SETUP DIAGNOSTIC")
    print("="*70)

    results = []

    # Run all checks
    results.append(("Brain Module Imports", check_brain_imports()))

    if not results[0][1]:
        print("\n⚠️ Cannot proceed - brain modules not available")
        print("Make sure you're running from the vision-engine directory")
        return

    results.append(("Ollama Server", check_ollama_running()))

    if results[1][1]:
        results.append(("Model Available", check_model_available()))

        if results[2][1]:
            results.append(("LLM Analyzer", test_llm_analyzer()))

    results.append(("Feature Aggregation", test_feature_aggregation()))

    # Summary
    print_header("DIAGNOSTIC SUMMARY")

    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} - {check_name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\nResults: {passed_count}/{total_count} checks passed")

    if passed_count == total_count:
        print("\n🎉 All checks passed! LLM system is working correctly.")
        print("\n📝 If you're still seeing generic alerts, the issue is:")
        print("   1. Vision engine not properly initialized")
        print("   2. Vision engine not detecting high enough risk scores")
        print("   3. Check streaming_server_integrated_optimized.py logs")
    else:
        print(f"\n⚠️ {total_count - passed_count} check(s) failed.")
        print("\n📝 Next steps:")
        for check_name, passed in results:
            if not passed:
                print(f"   - Fix: {check_name}")

    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
