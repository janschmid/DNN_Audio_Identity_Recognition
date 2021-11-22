from posixpath import split
from pytube import YouTube as YouTube
import os
from pydub import AudioSegment
from data_splitter import split_file

root_dir = file_path = os.path.dirname(os.path.realpath(__file__))
outputDir = os.path.join(root_dir, "youtubeDl")
link = "https://youtu.be/-2AxKyNAaM0"
yt = YouTube(link)
stream = yt.streams.get_highest_resolution()
os.makedirs(outputDir, exist_ok=True)
fileName = yt.title.lower().replace(' ', '_')+".mp4"
stream.download(outputDir, filename=fileName)

filePath = os.path.join(outputDir, fileName)
audioSegment = AudioSegment.from_file(filePath, format="mp4")
audioSegment.export(os.path.join(filePath.replace(".mp4", ".wav")), format="wav")
targetDir = os.path.join(outputDir+"splitted_100ms_chunks")
os.makedirs(targetDir, exist_ok=True)
split_file(filePath.replace(".mp4", ".wav"), 100, targetDir)