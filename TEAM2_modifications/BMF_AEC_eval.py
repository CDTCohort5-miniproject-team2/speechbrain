from scipy.io import wavfile
import sys
sys.path.append("../") 
import pysepm

fs, clean_speech = wavfile.read('')
fs, noisy_speech = wavfile.read('')
fs, enhanced_speech = wavfile.read('')

#---------------calculate STOI------------------

pysepm.stoi(clean_speech, noisy_speech, fs)
pysepm.stoi(clean_speech, enhanced_speech, fs)

#---------------calculate PESQ------------------

pysepm.pesq(clean_speech, noisy_speech, fs)
pysepm.pesq(clean_speech, enhanced_speech, fs)

#---------------calculate composite-------------

pysepm.composite(clean_speech, noisy_speech, fs)
pysepm.composite(clean_speech, enhanced_speech, fs)

