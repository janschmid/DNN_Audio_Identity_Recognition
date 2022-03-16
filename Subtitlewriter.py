import cv2
import numpy as np
import time


vid_dir = "/Users/NathanDurocher/Documents/Current_Classes/DNN/DNN_VIdeo.mp4"

cap = cv2.VideoCapture(vid_dir)
ret, frame = cap.read()

frame_idx = 12
frame_rate = 25 # Hz
frame_period = 1/frame_rate # secs
list_idx = 0
list_freq = 1 # secs

font = cv2.FONT_HERSHEY_SIMPLEX
colour = (0, 0, 255)
thickness = 2
org = (600, 560) 
class_list = np.loadtxt("WhichPodcaster_split_1s_chunks_RESNET_n_lable_3 _Mon_13_Dec_2021_12_50_21_test.csv")
names = ["Both", "Esben", "Peter"]

w, h, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, frame_rate, (h, w))

# while(cap.isOpened()):
while(list_idx < len(class_list)-1):
    ret, frame = cap.read()
    if ret==True:
        if frame_idx*frame_period > list_idx*list_freq:
            list_idx += 1
            person_class = class_list[list_idx]
            Text = names[int(person_class)]

        cv2.putText(frame, Text, org, font, 1, colour, thickness)
        out.write(frame)
        # cv2.imshow("Written", frame)
        #
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        break
    frame_idx += 1
    # time.sleep(0.05)

# Release everything if job is finished
print("Number of frames: %d, Number of Labels %d" % (frame_idx, list_idx+1))
cap.release()
out.release()
