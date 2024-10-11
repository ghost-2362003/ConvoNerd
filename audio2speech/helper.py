'''#calling subprocess to convert video to audio                   
## This is not needed as we are already providing audio file ##
command = "ffmpeg -i demo.mp4 -ab 160k -ar 44100 -vn audio.wav"
subprocess.call(command, shell=True)'''

import torch
import zipfile
import torchaudio
from glob import glob

device = torch.device('cpu')

model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       language='en',
                                       device=device)

(read_batch, split_into_batches,read_audio, prepare_model_input) = utils  

test_files = glob('audio2speech/media/audio_1.wav')
batches = split_into_batches(test_files)
input = prepare_model_input(read_batch(batches[0]),
                            device=device)

out_lst = []
output = model(input)
for out in output:
    out_lst.append(decoder(out.cpu()))
    
print("printing out_lst ...")
print(out_lst)