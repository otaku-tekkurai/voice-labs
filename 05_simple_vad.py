import soundfile as sf
import numpy as np

# Load audio
data, sr = sf.read("samples/sample_16k.wav")

# Frame settings
frame_ms = 20
frame_size = int(sr * frame_ms / 1000)

# VAD threshold (in dB) - adjust based on your audio
SILENCE_THRESHOLD_DB = -35  # Below this = silence

def get_frame_db(frame):
    rms = np.sqrt(np.mean(frame**2))
    return 20 * np.log10(rms + 1e-10)  # Avoid log(0)

# Analyze all frames
print("Frame analysis (S=silence, V=voice):\n")

results = []
for i in range(len(data) // frame_size):
    start = i * frame_size
    frame = data[start:start + frame_size]
    db = get_frame_db(frame)
    
    is_voice = db > SILENCE_THRESHOLD_DB
    results.append(is_voice)

# Print visual representation (50 frames per line)
timeline = ""
for i, is_voice in enumerate(results):
    timeline += "█" if is_voice else "░"
    if (i + 1) % 50 == 0:
        time_sec = (i + 1) * frame_ms / 1000
        print(f"{timeline} {time_sec:.1f}s")
        timeline = ""

# Print remaining
if timeline:
    time_sec = len(results) * frame_ms / 1000
    print(f"{timeline.ljust(50)} {time_sec:.1f}s")
 
# Statistics
voice_frames = sum(results)
silence_frames = len(results) - voice_frames
print(f"\nTotal frames: {len(results)}")
print(f"Voice frames: {voice_frames} ({voice_frames/len(results)*100:.1f}%)")
print(f"Silence frames: {silence_frames} ({silence_frames/len(results)*100:.1f}%)")