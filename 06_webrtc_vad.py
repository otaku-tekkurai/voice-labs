import soundfile as sf
import numpy as np
import webrtcvad

# Load audio
data, sr = sf.read("samples/sample_16k.wav")

# WebRTC VAD requirements:
# - Sample rate must be 8000, 16000, or 32000 Hz ✓
# - Frame must be 10, 20, or 30 ms ✓
# - Audio must be 16-bit PCM (we need to convert)

# Convert float64 to int16
data_int16 = (data * 32767).astype(np.int16)

# Create VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  # 0=least aggressive, 3=most aggressive

print("VAD Modes:")
print("  0 = Least aggressive (more false positives, catches soft speech)")
print("  1 = Low aggressive")
print("  2 = Medium aggressive (balanced) ← we're using this")
print("  3 = Most aggressive (more false negatives, stricter)")
print()

# Frame settings
frame_ms = 20
frame_size = int(sr * frame_ms / 1000)  # Number of samples per frame
num_frames = len(data_int16) // frame_size

# Analyze
results = []
for i in range(len(data_int16) // frame_size):
    start = i * frame_size
    frame = data_int16[start:start + frame_size]

    # Convert to bytes (webrtcvad needs bytes, not numpy array)
    frame_bytes = frame.tobytes()

    is_speech = vad.is_speech(frame_bytes, sr)
    results.append(is_speech)
    
# Print visual
print("WebRTC VAD Analysis:\n")
timeline = ""
for i, is_voice in enumerate(results):
    timeline += "█" if is_voice else "░"
    if (i + 1) % 50 == 0:
        time_sec = (i + 1) * frame_ms / 1000
        print(f"{timeline} {time_sec:.1f}s")
        timeline = ""

if timeline:
    print(f"{timeline.ljust(50)} {len(results) * frame_ms / 1000:.1f}s")

voice_frames = sum(results)
print(f"\nVoice: {voice_frames} frames ({voice_frames/len(results)*100:.1f}%)")
print(f"Silence: {len(results)-voice_frames} frames ({(len(results)-voice_frames)/len(results)*100:.1f}%)")