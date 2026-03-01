"""
Send Driver Alert to Backend
============================

Simple helper to POST driver alerts when risk >= 0.85
"""

import requests
import time


def send_driver_alert(
    track_id: int,
    risk_score: float,
    camera_id: str = "platform_3_camA",
    distance_from_edge: float = None,
    backend_url: str = "http://localhost:8000"
):
    """
    Send driver alert to backend API

    Args:
        track_id: Person track ID
        risk_score: Risk score (0-1)
        camera_id: Camera identifier
        distance_from_edge: Distance from platform edge in pixels
        backend_url: Backend API URL

    Returns:
        bool: True if successful, False otherwise

    Example:
        >>> if risk_score >= 0.85:
        >>>     send_driver_alert(
        >>>         track_id=42,
        >>>         risk_score=0.92,
        >>>         camera_id="platform_3_camA",
        >>>         distance_from_edge=25
        >>>     )
    """

    # Only send if risk >= 0.85
    if risk_score < 0.85:
        return False

    payload = {
        "track_id": track_id,
        "risk_score": risk_score,
        "camera_id": camera_id,
        "distance_from_edge": distance_from_edge,
        "timestamp": int(time.time() * 1000)  # milliseconds
    }

    try:
        response = requests.post(
            f"{backend_url}/api/driver-alert",
            json=payload,
            timeout=2.0
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✓ Driver alert sent: Track {track_id}, Risk {risk_score:.2f}")
                return True
            else:
                print(f"⚠ Alert below threshold: {result.get('message')}")
                return False
        else:
            print(f"✗ Driver alert failed: HTTP {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print(f"✗ Driver alert timeout (backend not responding)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Driver alert failed (backend not running)")
        return False
    except Exception as e:
        print(f"✗ Driver alert error: {e}")
        return False


# Example usage
if __name__ == "__main__":
    print("Testing driver alert system...\n")

    # Test scenarios
    scenarios = [
        (0.75, "Below threshold - should not alert"),
        (0.87, "Critical - should alert driver"),
        (0.96, "Emergency - should alert driver")
    ]

    for risk, description in scenarios:
        print(f"Testing: {description} (risk={risk:.2f})")
        success = send_driver_alert(
            track_id=42,
            risk_score=risk,
            camera_id="platform_3_camA",
            distance_from_edge=30
        )
        print(f"  Result: {'✓ Sent' if success else '✗ Not sent'}\n")
        time.sleep(1)

    print("✓ Test complete!")
