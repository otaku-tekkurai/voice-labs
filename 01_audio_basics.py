import soundfile as sf

# Load the audio file
data, samplerate = sf.read('samples/sample.aiff')

# Basic properties
print(f"Sample Rate: {samplerate} Hz")
print(f"Total Samples: {len(data)}")
print(f"Duration: {len(data) / samplerate:.2f} seconds")
print(f"Data Type: {data.dtype}")
print(f"Shape: {data.shape}")

if len(data.shape) == 1:
    print("Channels: Mono")
else:
    print(f"Channels: {data.shape[1]} (Stereo)" if data.shape[1] == 2 else f"Channels: {data.shape[1]}")
    
# sample values (first 10 samples)
print(f"\nFirst 10 samples: {data[:10]}")
print(f"Min value: {data.min():.4f}")
print(f"Max value: {data.max():.4f}")