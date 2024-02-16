from asteroid.models import BaseModel
import soundfile as sf
import librosa
import librosa.display
import sys

# 'from_pretrained' automatically uses the right model class (asteroid.models.DPRNNTasNet).
demo_audio_fpath = sys.argv[1]
model = BaseModel.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean")

# You can pass a NumPy array:
mixture, sr = sf.read(demo_audio_fpath, dtype="float32", always_2d=True)

# Resample audiofile if necessary
if sr != 8000:
    mixture = librosa.resample(mixture, orig_sr=sr, target_sr=8000)

# Soundfile returns the mixture as shape (time, channels), and Asteroid expects (batch, channels, time)
mixture = mixture.transpose()
mixture = mixture.reshape(1, mixture.shape[0], mixture.shape[1])
model.separate(mixture, force_overwrite = True)

# NOTE: soundfile needs to be Mono with a sampling freq of 8000Hz (can change the pretrained model to make it 16KHz instead)