
  | Scenario                            | Recommended
             |
  |-------------------------------------|------------------------------------------------------------
  -----------|
  | "User stopped talking, send to STT" | WebRTC mode 2-3 + silence duration threshold (e.g., 500ms
  of silence) |
  | "Trim silence from recording"       | Simple energy VAD
             |
  | "Detect interruption during TTS"    | WebRTC mode 1-2 (catch early)
             |

  ---
  Ready for the next topic. What would you like?

  1. Noise reduction - Add background noise to your sample, then remove it
  2. Move to STT/TTS patterns - How streaming transcription and synthesis work internally

  ---
  Reminder: We still need to cover STT/TTS integration patterns and interruption/turn-taking 
  mechanics.