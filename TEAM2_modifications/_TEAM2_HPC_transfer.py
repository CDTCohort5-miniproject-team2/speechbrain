import operator
from stat import ST_CTIME
import os, sys, time
import pysftp

# Specify your HPC server details

def send_recive(hpc_username, audio_file):

    hpc_audio_directory = f'ENTER DIRECTORY FOR AUDIO FILES ON HPC{audio_file}'
    local_audio_file = f'ENTER LOCAL DIRECTORY OF AUDIO FILE{audio_file}'

    name, extension = os.path.splitext(audio_file)
    transcript_file = f"{name}_transcript.json"
    print(transcript_file)

    hpc_transcript_directory = f'ENTER DIRECTORY FOR TRANSCRIPT FILES ON HPC {transcript_file}'
    local_transcript_directory = f'ENTER LOCAL DIRECTORY FOR TRANSCRIPT FILES {transcript_file}'

    with pysftp.Connection('ENTER HPC HOSTNAME', username=hpc_username,private_key='ENTER PATH TO PRIVATE KEY FILE') as sftp:

        sftp.put(local_audio_file, hpc_audio_directory)      
        print("waiting for answer....")
        while True:
            if sftp.exists(hpc_transcript_directory):
                sftp.get(hpc_transcript_directory,local_transcript_directory)
                break
        print("answer found!")
        with open(local_transcript_directory) as t:
            transcript = t.read()
        return transcript

if __name__ == '__main__':
    audio_file = sys.argv[1]
    hpc_username = 'ENTER USERNAME'

    transcript = send_recive(hpc_username, audio_file)
    print(transcript)