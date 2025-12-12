# Voice Agent Architecture & Audio Fundamentals

> This document captures the learning journey for building voice agents.

---

## How to Continue This Conversation

Copy-paste the prompt below to resume on another system:

```
I'm continuing a learning session about voice agent architecture.

Read this document first:
/Users/otakutekkurai/repos/claims/voice-lab/docs/01-voice-agent-architecture.md

**Where we left off:**
- Completed Exercises 1-5 (audio basics, metrics, resampling, frames, simple VAD)
- About to run Exercise 6: WebRTC VAD

**Remaining topics to cover:**
1. Noise reduction examples (noisereduce, RNNoise, DeepFilterNet)
2. Audio streaming examples (WebSocket, real-time mic)
3. STT/TTS integration patterns (how they work internally)
4. Interruption & turn-taking mechanics

**How we're working:**
- You explain concepts and show code
- I type the code myself to learn
- You do NOT write files directly (except docs)
- Remind me at the end of each response about STT/TTS and interruption topics

Let's continue from where we left off.
```

---

## Table of Contents

1. [Real-Time Voice Agent Architecture](#real-time-voice-agent-architecture)
2. [Audio Fundamentals](#audio-fundamentals)
3. [Audio Processing Libraries](#audio-processing-libraries)
4. [Hands-On Exercises](#hands-on-exercises)
5. [Next Topics](#next-topics)

---

## Real-Time Voice Agent Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                  │
│  [Phone/SIP] ─── [WebRTC Browser] ─── [Mobile App] ─── [IoT Device] │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ Audio Stream (bidirectional)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MEDIA GATEWAY                                    │
│  • Twilio / Vonage / Amazon Connect / FreeSWITCH                    │
│  • WebSocket/SIP signaling                                          │
│  • Audio codec handling (Opus, G.711, PCM)                          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ Audio chunks (20-100ms frames)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   VOICE ORCHESTRATOR                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   STT Engine │  │ Agent/LLM    │  │  TTS Engine  │               │
│  │  (Streaming) │──│   Brain      │──│  (Streaming) │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  • Turn detection (VAD - Voice Activity Detection)                  │
│  • Interruption handling                                            │
│  • State management                                                 │
│  • Latency optimization                                             │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BACKEND SERVICES                                  │
│  [Agent Logic] [Knowledge Base] [APIs] [Database] [Analytics]       │
└─────────────────────────────────────────────────────────────────────┘
```

### Speech Pipeline

**Inbound (User → Agent):**
```
Audio Input → VAD → STT (streaming) → Transcript → Agent
     │                    │
     │              Interim results
     │              (for responsiveness)
     ▼
 Noise reduction
 Echo cancellation
```

**Outbound (Agent → User):**
```
Agent Response → TTS (streaming) → Audio chunks → Media Gateway → User
                      │
                 First byte latency
                 critical (<300ms goal)
```

### STT Options

| Provider | Latency | Streaming | Best For |
|----------|---------|-----------|----------|
| Deepgram | ~100ms | Yes | Low latency, accuracy |
| Google STT | ~200ms | Yes | Multi-language |
| Azure Speech | ~150ms | Yes | Enterprise |
| Whisper (self-hosted) | ~500ms+ | Limited | Cost control |
| AssemblyAI | ~200ms | Yes | Features (diarization) |

### TTS Options

| Provider | Latency | Quality | Notes |
|----------|---------|---------|-------|
| ElevenLabs | ~200ms | Excellent | Voice cloning |
| OpenAI TTS | ~300ms | Good | Simple API |
| Azure Neural | ~150ms | Very Good | Enterprise |
| Cartesia | ~100ms | Good | Ultra-low latency |
| PlayHT | ~200ms | Good | Voice cloning |

### Voice Orchestration Patterns

**Pattern A: Sequential (Simple)**
```
STT complete → LLM → TTS → Play
                │
         High latency (2-4s)
```

**Pattern B: Streaming Pipeline (Recommended)**
```
STT interim → LLM streaming → TTS streaming → Play chunks
    │              │               │
    └──────────────┴───────────────┴── Parallel processing
                                        (~500ms to first audio)
```

**Pattern C: Full Duplex with Interruption**
```
┌─────────────────────────────────────┐
│     Continuous audio monitoring     │
│            ┌─────────┐              │
│  User ────►│   VAD   │──► Interrupt │
│  Audio     └─────────┘    Handler   │
│                              │      │
│                              ▼      │
│                        Stop TTS     │
│                        Process new  │
└─────────────────────────────────────┘
```

### Integration with LLM Agents

**A. Direct LLM Integration:**
```python
# Simple: Voice → STT → LLM API → TTS → Voice
voice_input → deepgram.transcribe() → openai.chat() → elevenlabs.speak()
```

**B. Agent Framework Integration:**
```python
# Complex: Voice orchestrator connects to your agent
voice_orchestrator ──WebSocket──► Your Agent Service
                                       │
                                       ├── LangGraph
                                       ├── CrewAI
                                       ├── Custom agents
                                       └── Tool calling
```

**C. Native Voice LLMs (Emerging):**
```
Audio → GPT-4o Realtime API → Audio
        (end-to-end, no STT/TTS)
```

### Key Technical Challenges

| Challenge | Solution |
|-----------|----------|
| **Latency** | Streaming everything, edge deployment, connection pooling |
| **Turn-taking** | VAD + silence detection + interruption handling |
| **Interruptions** | Bidirectional streaming, cancel TTS mid-sentence |
| **Context** | Maintain conversation state, handle repairs ("no, I said...") |
| **Noise/Quality** | Pre-processing, noise reduction, echo cancellation |
| **Scaling** | Stateful connections = harder. Use sticky sessions or state externalization |

### Deployment Architectures

**Option A: Managed Platform (Fastest to deploy)**
```
Twilio/Vonage ←→ Retell.ai / Vapi / Bland.ai ←→ Your Agent API
                      │
              Handles all voice infra
```

**Option B: Cloud Native (More control)**
```
┌──────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                     │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  Ingress    │  │   Voice     │  │   Agent     │      │
│  │  (WebSocket)│──│ Orchestrator│──│  Service    │      │
│  │  + Load Bal │  │  (Stateful) │  │ (Stateless) │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│         │                │                │              │
│         │         ┌──────┴──────┐         │              │
│         │         │   Redis     │         │              │
│         │         │(Session/Pub-Sub)      │              │
│         │         └─────────────┘         │              │
└─────────┼───────────────────────────────────────────────┘
          │
    Twilio/SIP Provider
```

### Recommended Stack for Production

**For getting started quickly:**
- **Media**: Twilio Voice + WebSocket streaming
- **Orchestration**: Pipecat or LiveKit Agents or Vocode
- **STT**: Deepgram (best latency/accuracy)
- **TTS**: Cartesia or ElevenLabs
- **Agent**: Your existing agent with HTTP/WebSocket interface

**Open-source orchestrators:**
| Tool | Description |
|------|-------------|
| **Pipecat** | Daily.co's framework, excellent for prototyping |
| **LiveKit Agents** | Full-featured, WebRTC native |
| **Vocode** | Python-based, good abstractions |
| **Jambonz** | Open-source CPaaS, self-hostable |

### Latency Budget (Target: <1s response time)

```
User stops speaking     0ms
VAD detects silence    +100ms
STT final transcript   +200ms (streaming helps)
Agent/LLM processing   +300ms (streaming helps)
TTS first byte         +200ms (streaming helps)
Network to user        +100ms
─────────────────────────────
Total to first audio   ~900ms ✓
```

---

## Audio Fundamentals

### How Sound Becomes Digital

```
Sound Wave (Analog)
     │
     ▼ Microphone converts to electrical signal
Electrical Signal
     │
     ▼ ADC (Analog-to-Digital Converter)
Digital Samples
     │
     ▼ Encoding
Audio File/Stream
```

### Key Audio Metrics

| Metric | What It Means | Typical Values |
|--------|---------------|----------------|
| **Sample Rate** | How many times per second we capture the signal | 8kHz (telephony), 16kHz (speech), 44.1kHz (music), 48kHz (video) |
| **Bit Depth** | Precision of each sample | 8-bit, 16-bit (standard), 24-bit (pro audio) |
| **Channels** | Mono vs Stereo | 1 (mono for voice), 2 (stereo) |
| **Bitrate** | Data per second (compressed audio) | 64-320 kbps |
| **Frame Size** | Chunk of samples processed together | 10ms, 20ms, 40ms |

**The Math:**
```
Uncompressed data rate = Sample Rate × Bit Depth × Channels

Example (telephony):
8000 samples/sec × 16 bits × 1 channel = 128,000 bits/sec = 128 kbps

Example (CD quality):
44100 × 16 × 2 = 1,411,200 bits/sec = 1.4 Mbps
```

### Audio Formats

**Uncompressed/Lossless:**
| Format | Extension | Description |
|--------|-----------|-------------|
| **PCM** | .raw, .pcm | Raw samples, no header. Must know sample rate/bit depth |
| **WAV** | .wav | PCM with header (contains metadata). Windows standard |
| **AIFF** | .aiff | Apple's equivalent of WAV |
| **FLAC** | .flac | Lossless compression (~50% size reduction) |

**Compressed/Lossy:**
| Format | Extension | Use Case |
|--------|-----------|----------|
| **MP3** | .mp3 | General audio, legacy |
| **AAC** | .aac, .m4a | Better than MP3, Apple/YouTube |
| **OGG Vorbis** | .ogg | Open source alternative |
| **Opus** | .opus | **Best for voice/real-time** (WebRTC default) |

**Telephony Codecs:**
| Codec | Bitrate | Quality | Latency |
|-------|---------|---------|---------|
| **G.711 (μ-law/A-law)** | 64 kbps | Good | Very low |
| **G.729** | 8 kbps | Acceptable | Low |
| **Opus** | 6-510 kbps | Excellent | Very low |
| **Speex** | 2-44 kbps | Good | Low |

**For Voice Agents: Opus is king** — adaptive bitrate, low latency, handles packet loss well.

### Audio Signal Metrics (Quality/Analysis)

| Metric | What It Measures |
|--------|------------------|
| **RMS (Root Mean Square)** | Average loudness/energy |
| **dB (Decibels)** | Logarithmic loudness scale |
| **dBFS** | Decibels relative to full scale. 0 dBFS = maximum |
| **SNR (Signal-to-Noise Ratio)** | Speech vs background noise. Higher = cleaner |
| **MOS (Mean Opinion Score)** | Perceived quality (1-5 scale) |

**Human Speech Characteristics:**
```
Fundamental frequency (pitch):
  - Male: 85-180 Hz
  - Female: 165-255 Hz

Intelligibility range: 300 Hz - 3400 Hz (telephony band)
Full speech range: 80 Hz - 8000 Hz
```

### dB Reference Scale

```
  0 dBFS ─┬─ Maximum (clipping/distortion)
          │
 -6 dBFS ─┼─ Half amplitude
          │
-20 dBFS ─┼─ Typical speech target
          │
-40 dBFS ─┼─ Quiet sounds
          │
-60 dBFS ─┼─ Near silence / noise floor
          │
   -∞ dB ─┴─ Complete silence
```

---

## Audio Processing Libraries

### Python Libraries Overview

| Task | Best Library | Notes |
|------|--------------|-------|
| **Audio I/O** | soundfile, librosa | soundfile faster, librosa more features |
| **Real-time mic/speaker** | pyaudio, sounddevice | sounddevice is simpler API |
| **Format conversion** | pydub, ffmpeg | pydub wraps ffmpeg |
| **Noise reduction** | DeepFilterNet, RNNoise | ML-based, very effective |
| **VAD (fast)** | webrtcvad | Low CPU, real-time |
| **VAD (accurate)** | silero-vad | ML-based, handles edge cases |
| **Speaker diarization** | pyannote-audio | Who spoke when |
| **Echo cancellation** | speexdsp | Standard AEC |
| **Spectral analysis** | librosa | FFT, mel spectrograms |
| **Quality metrics** | pesq, pystoi | Industry standard |

### What is Streaming?

**Batch Processing:**
```
[Record complete audio] → [Send entire file] → [Process all] → [Return result]
         3 seconds              upload            2 seconds        response

Total latency: 5+ seconds before you get anything
```

**Streaming Processing:**
```
[Audio chunk 1] → [Process] → [Partial result 1]    ← 100ms
[Audio chunk 2] → [Process] → [Partial result 2]    ← 200ms
[Audio chunk 3] → [Process] → [Partial result 3]    ← 300ms
...
```

### Streaming Protocols

| Protocol | Use Case | Bidirectional | Notes |
|----------|----------|---------------|-------|
| **WebSocket** | Web/general | Yes | Most common for voice agents |
| **gRPC** | Backend services | Yes | Efficient, typed, streaming native |
| **RTP/RTCP** | VoIP/telephony | Yes | UDP-based, low latency |
| **WebRTC** | Browser P2P | Yes | Built-in echo cancel, NAT traversal |

---

## Hands-On Exercises

### Project Setup

```bash
cd /Users/otakutekkurai/repos/claims
uv init voice-lab
cd voice-lab
```

### Dependencies Added

```bash
uv add soundfile
uv add librosa
uv add webrtcvad
```

### Exercise 1: Audio Basics (`01_audio_basics.py`)

Read an audio file and inspect its properties:

```python
import soundfile as sf

# Load the audio file
data, sample_rate = sf.read("samples/sample.aiff")

# Basic properties
print(f"Sample Rate: {sample_rate} Hz")
print(f"Total Samples: {len(data)}")
print(f"Duration: {len(data) / sample_rate:.2f} seconds")
print(f"Data Type: {data.dtype}")
print(f"Shape: {data.shape}")

# Check mono vs stereo
if len(data.shape) == 1:
    print("Channels: 1 (Mono)")
else:
    print(f"Channels: {data.shape[1]}")

# Sample values
print(f"\nFirst 10 samples: {data[:10]}")
print(f"Min value: {data.min():.4f}")
print(f"Max value: {data.max():.4f}")
```

### Exercise 2: Audio Metrics (`02_audio_metrics.py`)

Calculate RMS and dB loudness:

```python
import soundfile as sf
import numpy as np

data, sample_rate = sf.read("samples/sample.aiff")

# RMS (Root Mean Square) - average loudness
rms = np.sqrt(np.mean(data ** 2))
print(f"RMS (average loudness): {rms:.4f}")

# Convert to decibels
db = 20 * np.log10(rms)
print(f"Loudness in dB: {db:.2f} dBFS")

# Peak in dB
peak_db = 20 * np.log10(max(abs(data.min()), abs(data.max())))
print(f"Peak in dB: {peak_db:.2f} dBFS")
```

### Exercise 3: Resampling (`03_resampling.py`)

Convert sample rates for different use cases:

```python
import soundfile as sf
import librosa

data, sr_original = sf.read("samples/sample.aiff")
print(f"Original: {sr_original} Hz, {len(data)} samples, {len(data)/sr_original:.2f}s")

# Resample to 16kHz (speech standard for STT)
data_16k = librosa.resample(data, orig_sr=sr_original, target_sr=16000)
print(f"Resampled: 16000 Hz, {len(data_16k)} samples, {len(data_16k)/16000:.2f}s")

# Resample to 8kHz (telephony)
data_8k = librosa.resample(data, orig_sr=sr_original, target_sr=8000)
print(f"Telephony: 8000 Hz, {len(data_8k)} samples, {len(data_8k)/8000:.2f}s")

# Save the 16kHz version
sf.write("samples/sample_16k.wav", data_16k, 16000)
print("\nSaved: samples/sample_16k.wav")
```

### Exercise 4: Audio Frames (`04_audio_frames.py`)

Understand how streaming works with frames:

```python
import soundfile as sf
import numpy as np

data, sr = sf.read("samples/sample_16k.wav")

# Frame parameters
frame_duration_ms = 20  # 20ms is standard for voice
frame_size = int(sr * frame_duration_ms / 1000)

print(f"Sample rate: {sr} Hz")
print(f"Frame duration: {frame_duration_ms} ms")
print(f"Frame size: {frame_size} samples")
print(f"Bytes per frame (16-bit): {frame_size * 2} bytes")

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
```

### Exercise 5: Simple VAD (`05_simple_vad.py`)

Energy-based voice activity detection:

```python
import soundfile as sf
import numpy as np

data, sr = sf.read("samples/sample_16k.wav")

frame_ms = 20
frame_size = int(sr * frame_ms / 1000)
SILENCE_THRESHOLD_DB = -35

def get_frame_db(frame):
    rms = np.sqrt(np.mean(frame ** 2))
    return 20 * np.log10(rms + 1e-10)

print("Frame analysis (S=silence, V=voice):\n")

results = []
for i in range(len(data) // frame_size):
    start = i * frame_size
    frame = data[start:start + frame_size]
    db = get_frame_db(frame)

    is_voice = db > SILENCE_THRESHOLD_DB
    results.append(is_voice)

# Print visual representation
timeline = ""
for i, is_voice in enumerate(results):
    timeline += "█" if is_voice else "░"
    if (i + 1) % 50 == 0:
        time_sec = (i + 1) * frame_ms / 1000
        print(f"{timeline} {time_sec:.1f}s")
        timeline = ""

if timeline:
    time_sec = len(results) * frame_ms / 1000
    print(f"{timeline.ljust(50)} {time_sec:.1f}s")

voice_frames = sum(results)
silence_frames = len(results) - voice_frames
print(f"\nTotal frames: {len(results)}")
print(f"Voice frames: {voice_frames} ({voice_frames/len(results)*100:.1f}%)")
print(f"Silence frames: {silence_frames} ({silence_frames/len(results)*100:.1f}%)")
```

### Exercise 6: WebRTC VAD (`06_webrtc_vad.py`)

Production-grade voice activity detection:

```python
import soundfile as sf
import numpy as np
import webrtcvad

data, sr = sf.read("samples/sample_16k.wav")

# Convert float64 to int16
data_int16 = (data * 32767).astype(np.int16)

# Create VAD
vad = webrtcvad.Vad()
vad.set_mode(2)  # 0=least aggressive, 3=most aggressive

print("VAD Modes:")
print("  0 = Least aggressive (catches soft speech)")
print("  1 = Low aggressive")
print("  2 = Medium aggressive (balanced) ← using this")
print("  3 = Most aggressive (stricter)")
print()

frame_ms = 20
frame_size = int(sr * frame_ms / 1000)

results = []
for i in range(len(data_int16) // frame_size):
    start = i * frame_size
    frame = data_int16[start:start + frame_size]
    frame_bytes = frame.tobytes()

    is_speech = vad.is_speech(frame_bytes, sr)
    results.append(is_speech)

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
```

---

## Next Topics

**Still to cover:**

1. **STT/TTS Integration Patterns**
   - How STT streaming works internally
   - How TTS streaming works internally
   - Integration code examples

2. **Interruption & Turn-Taking Mechanics**
   - How to detect user interruptions
   - Canceling TTS mid-sentence
   - Handling overlapping speech
   - Turn-taking state machines

3. **Noise Reduction Examples**
   - noisereduce library
   - RNNoise / DeepFilterNet

4. **Audio Streaming Examples**
   - WebSocket audio streaming
   - Real-time microphone capture

---

## Commands Reference

```bash
# Run any exercise
uv run python 01_audio_basics.py
uv run python 02_audio_metrics.py
uv run python 03_resampling.py
uv run python 04_audio_frames.py
uv run python 05_simple_vad.py
uv run python 06_webrtc_vad.py
```

---

*Document created: 2024-12-12*
*Last updated: 2024-12-12*
