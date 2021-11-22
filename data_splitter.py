from pydub import AudioSegment
import os
from custom_dataset import AudioDataset

def split_file(fileDir, chunkLength, targetDir):
    newAudio = AudioSegment.from_wav(fileDir)
    for i in range(0, int(newAudio.duration_seconds*1000), chunkLength):
            chunk = newAudio[i:i+chunkLength]
            if(chunk.duration_seconds==0.0):
                continue
            os.makedirs(targetDir, exist_ok=True)
            fileName = os.path.basename(file).split('.')[0]+"_chunk_{0}.".format(i)+os.path.basename(file).split('.')[1]
            chunk.export(os.path.join(targetDir, fileName), format="wav") #Exports to a wav file in the current path.


root_dir = file_path = os.path.dirname(os.path.realpath(__file__))
origDataset = "dataset"
datasetDir = os.path.join(root_dir, origDataset)
chunkLength = 100 #Works in milliseconds
datasetSplittedDir = os.path.join(root_dir, "{0}_{1}msec_chunks".format(origDataset, chunkLength))
os.makedirs(datasetSplittedDir, exist_ok=True)
# gdd.download_file_from_google_drive('1Plh2FwwShj9fq-lCZaFqT04D7wi5F5RN', os.path.join(datasetDir, "podcast_audio.zip"), unzip=True)

data = AudioDataset(datasetDir,500)
for file in data.img_paths:
    split_file(file, chunkLength, datasetSplittedDir)
    # newAudio = newAudio[t1:t2]
    # newAudio.export('newSong.wav', format="wav") #Exports to a wav file in the current path.