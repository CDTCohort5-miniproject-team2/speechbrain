import torch
import numpy as np
import sounddevice as sd
import queue
import time
from datetime import datetime
import scipy

class MicListener:
    def __init__(self, sampling_rate=16000, n_channels=1, callback_chunk=512,
                 max_buffer_dur=10.0, vad_threshold=0.5, await_silence_dur=3.0):
        self.sampling_rate = sampling_rate
        # Silero VAD is trained on audio chunks of 30ms (480 samples given sr=16kHz)
        self.callback_chunk = callback_chunk
        self.max_buffer_size = max_buffer_dur * sampling_rate
        self.n_channels = n_channels
        self.await_silence_size = await_silence_dur * sampling_rate / callback_chunk

        self.collected_audio = np.array([], dtype=np.float32)
        self.partially_transcribed_length = 0
        self.vad_flag_seq = []
        self.vad_threshold = vad_threshold
        self.previous_chunk = None
        self.n_silent_chunks = 0

        self.q = queue.Queue()
        self.input_stream_start_time = 0.0
        self.stream_dur = 0.0
        self.input_stream_closing = False

        # Voice Activity Detection model setup
        self.vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = vad_utils

    def transcribe(self, wavfile):

        transcribed_text = ""
        return transcribed_text

    def _input_audio_callback(self, wav_array, frame_count, time_info, status_flag):
        if status_flag:
            print(status_flag)
        if not self.input_stream_closing:
            self.q.put(wav_array.copy())

    def save_wav(self, wav_array, use_current_timestamp=True, filename_prefix="demo", output_dir="demo_results"):
        if use_current_timestamp:
            filename = f"{output_dir}/{filename_prefix}_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        else:
            filename = f"{output_dir}/{filename_prefix}.wav"
        scipy.io.wavfile.write(filename, self.sampling_rate, wav_array)

    def start_input_stream(self, stream_dur):
        input_stream = sd.InputStream(samplerate=self.sampling_rate, blocksize=self.callback_chunk, channels=self.n_channels,
                                      callback=self._input_audio_callback)
        self.stream_dur = stream_dur

        with input_stream:
            print("I'm listening...")
            self.input_stream_start_time = time.time()
            while True:
                new_chunk = self.q.get()
                new_chunk = np.squeeze(new_chunk)

                speech_prob = self.vad_model(torch.tensor(new_chunk), self.sampling_rate)
                # print(f"\rSpeech probability: {speech_prob}", end="")
                self.vad_flag_seq.append((speech_prob >= self.vad_threshold))

                if speech_prob >= self.vad_threshold:
                    if self.n_silent_chunks > 0:
                        # including the previous (non-speech) chunk that precedes the current speech chunk
                        self.collected_audio = np.concatenate((self.collected_audio, self.previous_chunk))
                    self.collected_audio = np.concatenate((self.collected_audio, new_chunk))
                    # reset continuous silent chunk counter
                    self.n_silent_chunks = 0
                else:
                    if self.n_silent_chunks == 0:
                        # including the current non-speech chunk that follows a previous speech chunk
                        self.collected_audio = np.concatenate((self.collected_audio, new_chunk))
                    self.n_silent_chunks += 1

                # storing current chunk in cache
                self.previous_chunk = new_chunk

                if (self.n_silent_chunks > self.await_silence_size) and (len(self.collected_audio) > self.sampling_rate / 10):
                    # send entire audio array for FULL transcription, then reset the collected_audio
                    padded_array = np.pad(self.collected_audio, (0, self.sampling_rate), mode="constant")
                    self.save_wav(padded_array)
                    self.collected_audio = np.array([], dtype=np.float32)
                    self.partially_transcribed_length = 0

                elif len(self.collected_audio)-self.partially_transcribed_length > self.max_buffer_size:
                    # TODO: send current array for PARTIAL transcription but keep concatenating
                    self.save_wav(self.collected_audio)
                    self.partially_transcribed_length = len(self.collected_audio)

                if (not self.input_stream_closing) and time.time() - self.input_stream_start_time >= self.stream_dur:
                    self.input_stream_closing = True

                if self.input_stream_closing and self.q.empty():
                    break

