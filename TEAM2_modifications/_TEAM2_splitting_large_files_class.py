import math, os
from pydub import AudioSegment
from scipy.io import wavfile

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename, format="wav")
        
    def multiple_split(self, secs_per_split):
        total_secs = math.ceil(self.get_duration())
        for i in range(0, total_secs, secs_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+secs_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_secs - secs_per_split:
                print('All splited successfully')
    
    def get_channel(self, channel_number):
        fs, data = wavfile.read(self.filepath)
        name, extension = os.path.splitext(self.filename)
        file_destination = f'{name}_channel_{channel_number}.wav'
        wavfile.write(file_destination, fs, data[:, (channel_number-1)])
        print(f'New audiofile saved for channel {channel_number} at {file_destination}')