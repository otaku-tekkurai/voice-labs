import soundfile as sf
import librosa

# Load original
data, sr_original = sf.read('samples/sample.aiff')
print(f"Original: {sr_original} Hz, {len(data)} samples, {len(data)/sr_original:.2f}s")

# Resample to 16kHz (speech standard for STT)
data_16k = librosa.resample(data, orig_sr=sr_original, target_sr=16000)
print(f"Resampled: 16000 Hz, {len(data_16k)} samples, {len(data_16k)/16000:.2f}s")

# Resample to 8kHz (telephony)
data_8k = librosa.resample(data, orig_sr=sr_original, target_sr=8000)
print(f"Telephony: 8000 Hz, {len(data_8k)} samples, {len(data_8k)/8000:.2f}s")

# Save the 16kHz version (what you'd send to STT)
sf.write("samples/sample_16k.wav", data_16k, 16000)
print("\nSaved: samples/sample_16k.wav")