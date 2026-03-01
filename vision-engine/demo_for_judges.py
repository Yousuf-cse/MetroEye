"""
MetroEye Demo for Judges
========================
Simulates edge detection and PA announcement trigger
"""

import time
from pa_announcement_system import PAAnnouncement
import os

print("="*60)
print("🎬 MetroEye PA Announcement Demo")
print("="*60)

# Initialize PA system
print("\n📡 Initializing PA system...")
pa = PAAnnouncement(
    elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
    use_elevenlabs=True,
    model="eleven_flash_v2_5"
)

print("\n" + "="*60)
print("🎭 SIMULATION: Person Near Platform Edge")
print("="*60)

# Simulate detection
print("\n[00:00] ✓ Person detected on platform")
print("[00:03] → Person walking toward edge...")
time.sleep(1)

print("[00:05] ⚠️  Person entered danger zone (95px from edge)")
print("           Timer started...")
time.sleep(1)

print("[00:06] ⚠️  Still near edge (85px) - 1 second")
time.sleep(1)

print("[00:07] ⚠️  Still near edge (80px) - 2 seconds")
time.sleep(1)

print("[00:08] ⚠️  Still near edge (75px) - 3 seconds")
time.sleep(1)

print("[00:09] ⚠️  Still near edge (70px) - 4 seconds")
time.sleep(1)

print("[00:10] 🚨 Still near edge (65px) - 5 seconds")
print("\n" + "="*60)
print("🔊 ALERT TRIGGERED! Person has been near edge for 5+ seconds")
print("="*60)

# Calculate simulated risk score
distance = 65  # pixels from edge
dwell_time = 5.2  # seconds
risk_score = 0.85  # High risk (0.80-0.94 = Level 2)

print(f"\n📊 Risk Analysis:")
print(f"   Distance from edge: {distance}px")
print(f"   Dwell time: {dwell_time}s")
print(f"   Risk Score: {risk_score:.2f} (0-1 scale)")
print(f"   Alert Level: LEVEL 2 - CRITICAL")

print(f"\n🔊 Playing PA announcement...")
print("="*60)

# Play Level 2 announcement (pre-cached, instant)
pa.play_level2_announcement()

print("\n" + "="*60)
print("✅ Demo Complete!")
print("="*60)
print("\n📋 Summary:")
print("   ✓ Person detected near edge")
print("   ✓ Tracked for 5+ seconds in danger zone")
print("   ✓ PA announcement triggered automatically")
print("   ✓ Security alerted to take action")
print("\n🎯 In production, this happens automatically for every camera!")
print("="*60)
