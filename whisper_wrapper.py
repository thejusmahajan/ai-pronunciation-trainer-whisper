# An enhansed wrapper removing (possible) whisper rules on silence processing.
import torch
from transformers import pipeline
from ModelInterfaces import IASRModel # 
from typing import Union
import numpy as np

class WhisperASRModel(IASRModel):
    def __init__(self, model_name="openai/whisper-base"):
        # Ensure device placement is handled appropriately if CUDA is available/desired
        # Using device_map="auto" or specifying device="cuda:0" if torch.cuda.is_available()
        # For simplicity, defaulting to CPU as per original structure if not specified
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            return_timestamps="word",
            # Consider adding chunk_length_s=30, stride_length_s=5 for long audio
            device=device
            )
        self._transcript = ""
        self._word_locations = []
        self.sample_rate = 16000 # Whisper models are trained on 16kHz audio

    def processAudio(self, audio: Union[np.ndarray, torch.Tensor]):
        # Ensure audio is a numpy array float32 as expected by the pipeline
        if isinstance(audio, torch.Tensor):
            # Ensure tensor is on CPU, detach from graph, convert to numpy
            audio_np = audio.detach().cpu().numpy()
        elif isinstance(audio, np.ndarray):
            audio_np = audio
        else:
            raise TypeError("Audio input must be a PyTorch Tensor or a NumPy array.")

        # Ensure the numpy array is float32
        if audio_np.dtype != np.float32:
             audio_np = audio_np.astype(np.float32)

        # If the input has 2 dimensions (e.g., [1, N]), take the first channel
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze() # Remove dimensions of size 1
            if audio_np.ndim > 1: # If still > 1D (e.g. stereo), take the first channel
                 audio_np = audio_np[0]

        # Pass the numpy array directly
        result = self.asr(audio_np) # Pass the processed numpy array

        self._transcript = result["text"]

        # Handle potential missing end timestamps
        processed_locations = []
        for word_info in result["chunks"]:
            start_ts_val = word_info["timestamp"][0]
            end_ts_val = word_info["timestamp"][1]

            # Ensure start_ts is always valid
            if start_ts_val is None:
                # I had issues with chrome where it abruptly through up an error during process 
                # Handle cases where even start_ts might be missing (less likely)
                 # Maybe skip the word or use the previous word's end time?
                 # For now, let's skip if start is None, or you could default to 0 or previous end.
                 print(f"Warning: Skipping word '{word_info['text']}' due to missing start timestamp.")
                 continue

            start_ts_samples = int(start_ts_val * self.sample_rate)

            # Check and handle None for end_ts
            if end_ts_val is not None:
                end_ts_samples = int(end_ts_val * self.sample_rate)
            else:
                # Default: Set end time 0.1 seconds after start time if end is missing
                print(f"Warning: Missing end timestamp for word '{word_info['text']}'. Using default duration.")
                end_ts_samples = start_ts_samples + int(0.1 * self.sample_rate)
                # Ensure end time is not before start time
                if end_ts_samples <= start_ts_samples:
                    end_ts_samples = start_ts_samples + self.sample_rate // 10 # Add a small buffer if start/end are too close

            processed_locations.append({
                "word": word_info["text"],
                "start_ts": start_ts_samples,
                "end_ts": end_ts_samples
            })

        self._word_locations = processed_locations


    def getTranscript(self) -> str:
        return self._transcript

    def getWordLocations(self) -> list:
        return self._word_locations
