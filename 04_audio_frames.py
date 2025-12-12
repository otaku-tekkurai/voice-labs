import soundfile as sf
import numpy as np

# Load 16KHz audio file
data, sr = sf.read('samples/sample_16k.wav')

# Frame parameters
frame_duration_ms = 20 # 20ms is standard for voice
frame_size = int(sr * frame_duration_ms / 1000) # samples per frame

print(f"Sample rate: {sr} Hz")
print(f"Frame duration: {frame_duration_ms} ms")
print(f"Frame size: {frame_size} samples")
print(f"Bytes per frame (16-bit): {frame_size * 2} bytes")

# How may frames in the audio?
total_frames = len(data) // frame_size
print(f"\nTotal complete frames: {total_frames}")
print(f"Total duration covered: {total_frames * frame_duration_ms / 1000:.2f}s")

# Extract first few frames
print("\n--- First 5 frames ---")
for i in range(5):
    start = i * frame_size
    end = start + frame_size
    frame = data[start:end]

    frame_rms = np.sqrt(np.mean(frame ** 2))
    frame_db = 20 * np.log10(frame_rms + 1e-10)

    print(f"Frame {i}: samples [{start:5d}:{end:5d}], RMS: {frame_rms:.4f}, dB: {frame_db:.1f}")