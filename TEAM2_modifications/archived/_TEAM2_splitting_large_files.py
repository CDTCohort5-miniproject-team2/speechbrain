import sys
from _TEAM2_splitting_large_files_class import SplitWavAudioMubin

audio_file = SplitWavAudioMubin(folder = sys.argv[1], filename = sys.argv[2])

#audio_file.get_channel(8) # Channel for one of the 5-wall-mics
#audio_file.get_channel(16) # Channel for the server close-talk mic

#audio_file.multiple_split(20)