from posixpath import split
from pytube import YouTube as YouTube
import os
from pydub import AudioSegment
from data_splitter import split_file
import shutil
import pathlib


root_dir = os.path.dirname(os.path.realpath(__file__))
outputDir = os.path.join(root_dir, "youtubeDl","3minClip" )
link = "https://www.youtube.com/watch?v=7Pf3VYfP-7I"
yt = YouTube(link)
stream = yt.streams.get_highest_resolution()
os.makedirs(outputDir, exist_ok=True)
fileName = yt.title.lower().replace(' ', '_')+".mp4"
stream.download(outputDir, filename=fileName)
# fileName="audioDownload_01_10_21.mp3"
inputFileExtension = pathlib.Path(fileName).suffix.split('.')[1]
filePath = os.path.join(outputDir, fileName)
audioSegment = AudioSegment.from_file(filePath, format=inputFileExtension)
audioSegment.set_channels(1)

wavFile = os.path.join(filePath.replace(inputFileExtension, "wav"))
# audioSegment.export(wavFile, format="wav")

audioSegment.export(wavFile, codec="pcm_s16le", format="wav", bitrate='256k', parameters=['-ar', '16000', '-ac', '1'])
targetDir = os.path.join(outputDir,  "split_1s_chunks/allData")
shutil.rmtree(targetDir, ignore_errors=True)
os.makedirs(targetDir, exist_ok=True)
split_file(filePath.replace(inputFileExtension, "wav"), 1000, targetDir)