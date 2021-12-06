import numpy as np
class_list = np.loadtxt("100ms_chunks_video.csv")

list_idx = 0

while(list_idx < len(class_list)-1):
    outputText = ""
    outputText.append("{0}\n"+list_idx)
    outputText.append()