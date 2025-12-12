import soundfile as sf
import numpy as np

# Load audio
data, sample_rate = sf.read("samples/sample.aiff")

# RMS (Root Mean Square) - average loudness
rms = np.sqrt(np.mean(data ** 2))
print(f"RMS (average loudness): {rms:.4f}")

# Convert to decibels (dB)
# dB is logarithmic - humans perceive loudness logarithmically
db = 20 * np.log10(rms)
print(f"Loudness in dB: {db:.2f} dBFS")

# Peak in dB
peak_db = 20 * np.log10(max(abs(data.min()), abs(data.max())))
print(f"Peak in dB: {peak_db:.2f} dBFS")