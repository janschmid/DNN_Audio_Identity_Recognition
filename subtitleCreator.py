import numpy as np
import csv
import datetime


def get_strftime(time):
    return time.strftime('%H:%M:%S,%f')[:-3]


def createSrtEntry(sequenceNumber, begin, end, content):
    srtEntry =""\
"{0}\r\n\
{1} --> {2}\r\n\
{3}\r\n\
\r\n\
".format(sequenceNumber, get_strftime(begin), get_strftime(end), content)
    return srtEntry

file = "/home/jschm20/DNN/Miniproject/output/WhichPodcaster_split_1s_chunks RESNET n_lab: 2  Mon, 13 Dec 2021 18:24:10_test.csv"
timeStep = 1#sec
file = np.loadtxt(file)
content=""
randomStartTime =  datetime.datetime(1990,1,1,0,0,0)
for i in range(len(file)):
    begin = randomStartTime+datetime.timedelta(seconds = i)
    end = randomStartTime+datetime.timedelta(seconds = i+1)
    if(file[i]==0):
        speaker="Esben"
    if(file[i])==1:
        speaker="Peter"
    if(file[i]==2):
        speaker="WTF???"
    content=content+createSrtEntry(i, begin, end, speaker)

file = open("test.srt", 'w')
file.write(content)
file.close()