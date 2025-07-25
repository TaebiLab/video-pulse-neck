import numpy as np
import pandas as pd
import cv2
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
import datetime
import random
import scipy.io
from pathlib import Path
from utils.general import increment_path
import skin_detector
import sys
import math
from tkinter import *
from tkinter import ttk
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from moviepy.editor import AudioFileClip


# Function to apply bandpass filter
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y


def takeinputs():
    global subject_id, breathing_hold
    window = Tk()
    window.title('Pre-processing video')
    window.geometry("400x250")
    
    def comboclick():
        global subject_id, breathing_hold
        subject_id = cb.get()
        
        if subject_id != 'Select': #and location != 'Select':
            radiobtn_val = v0.get()
            if radiobtn_val == 1:
                breathing_hold = "EndExhalation"
            elif radiobtn_val == 2:
                breathing_hold = "EndInhalation"            
            window.destroy()        
    
    options = ['P01-221021_01','P02-221026_01','P03-221025_01','P04-221104_01','P05-221104_01','P07-221027_01',
               'P08-221027_01','P09-221027_01','P10-221027_01','P11-221028_01','P12-221028_01','P13-221103_01',
               'P14-221103_01','P15-221103_01']    
    # Radio button
    v0=IntVar()
    v0.set(1)
    r1=Radiobutton(window, text="EndExhalation", variable=v0, value=1)
    r2=Radiobutton(window, text="EndInhalation", variable=v0, value=2)    
    r1.place(x=50,y=25)
    r2.place(x=160, y=25)    
    # Subject selection
    label = ttk.Label(window, text = "Select a Subject :")
    label.place(x=55, y=90)    
    cb = ttk.Combobox(window, values=options)
    cb.set('Select')
    cb.pack()
    cb.place(x=180, y=90)    
    # Sumbmit Button
    submit = Button(window, text='Submit', command=comboclick)
    submit.pack()
    submit.place(x=270, y=180)
    
    window.mainloop()
    return subject_id, breathing_hold

def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# calculate displacement in x and y direction form Franeback algorithm
def calculate_displacement(magnitude, angle_radians):
    # # Convert angle from degrees to radians
    # angle_radians = np.radians(angle_degrees)
    
    # Calculate x-component 
    delta_x = magnitude * np.cos(angle_radians)
    
    # Calculate y-component 
    delta_y = magnitude * np.sin(angle_radians)
    
    return delta_x, delta_y


def plot_signal(signal_x, signal_y, fps):
    # plot signals
    t, dt = np.linspace(0, len(signal_y)/fps, len(signal_y), retstep=True)

    # Calculate derivatives
    #dt = t[1]-t[0]
    # Velocity
    dxdt = np.gradient(signal_x)/dt
    dydt = np.gradient(signal_y)/dt
    # Acceleration
    dx2dt = np.gradient(dxdt)/dt
    dy2dt = np.gradient(dydt)/dt

    fig, (ax1, dx1, dx2, ay1, dy1, dy2) = plt.subplots(6, sharex=True)
    fig.suptitle('x and y plot vs time')
    ax1.plot(t, signal_x)
    ax1.set(ylabel='x displacement')
    dx1.plot(t, dxdt)
    dx1.set(ylabel='Velocity')
    dx2.plot(t, dx2dt)
    dx2.set(ylabel='Acceleration')
    ay1.plot(t, signal_y)
    ay1.set(ylabel='y displacement')
    dy1.plot(t, dydt)
    dy1.set(ylabel='Velocity')
    dy2.plot(t, dy2dt)
    dy2.set(xlabel ='time', ylabel='Acceleration')
    plt.show()
    
    

scale_percent = 25
visualize = False
padding = 20   # padding is used for the serach area qr region from the current frame. After that the template match will search with in the region for the template

# EndExhalation Tap Sound
EndExhalation_tap = {'P01-221021_01':{'start':6.8182,'end':19.195},
                     'P02-221026_01':{'start':5.1462,'end':19.1},
                     'P03-221025_01':{'start':4.4046,'end':18.407},
                     'P04-221104_01':{'start':'None','end':'None'},
                     'P05-221104_01':{'start':3.3896,'end':19.5},
                     'P07-221027_01':{'start':3.4812,'end':16.997},
                     'P08-221027_01':{'start':5.112,'end':19.502},
                     'P09-221027_01':{'start':9.2035,'end':21.023},
                     'P10-221027_01':{'start':6.8129,'end':20.428},
                     'P11-221028_01':{'start':6.2696,'end':20.041},
                     'P12-221028_01':{'start':7.1966,'end':20.981},
                     'P13-221103_01':{'start':1.593,'end':16.228},
                     'P14-221103_01':{'start':2.0915,'end':16.946},
                     'P15-221103_01':{'start':2.8997,'end':19.323}}


# EndInhalation Tap Sound
EndInhalation_tap = {'P01-221021_01':{'start':6.5817,'end':20.872},
                     'P02-221026_01':{'start':7.0887,'end':20.431},
                     'P03-221025_01':{'start':5.699,'end':19.694},
                     'P04-221104_01':{'start':1.2293,'end':16.939},
                     'P05-221104_01':{'start':2.6135,'end':19.219},
                     'P07-221027_01':{'start':7.4324,'end':21.3},
                     'P08-221027_01':{'start':5.9152,'end':17.382},
                     'P09-221027_01':{'start':7.4716,'end':18.918},
                     'P10-221027_01':{'start':3.5287,'end':16.661},
                     'P11-221028_01':{'start':7.2688,'end':19.83},
                     'P12-221028_01':{'start':4.9293,'end':19.599},
                     'P13-221103_01':{'start':1.4729,'end':18.796},
                     'P14-221103_01':{'start':0.21583,'end':16.331}}

plt.close('all')

subject_id, breathing_hold = takeinputs()

path = "\Data\human_subjects"

filename = subject_id+'_'+breathing_hold+'.MOV'

video_file = os.path.join(path,subject_id,"Pre-processed",subject_id+'_'+breathing_hold,filename)
if os.path.exists(video_file):
    rawVideo = cv2.VideoCapture(video_file)
else:
    video_file = os.path.join(path,subject_id,filename)
    rawVideo = cv2.VideoCapture(video_file)
    print("Preprocessed video file not exists. Reading video form original directory")
    
if not rawVideo.isOpened():
    print("Error opening video file.")
    sys.exit()
    
#find video frame rate
fps = rawVideo.get(cv2.CAP_PROP_FPS)
print("Video: {}, frame rate: {}".format(filename,fps))
    
n_frame = int(rawVideo.get(cv2. CAP_PROP_FRAME_COUNT)) 


frames = np.empty((n_frame,),dtype=np.ndarray)
n_frame = int(rawVideo.get(cv2. CAP_PROP_FRAME_COUNT)) 

print("Video: {}, frame rate: {}, n_frame: {}".format(filename,fps,n_frame))


# Starting frame before the starting sound 
if breathing_hold == "EndExhalation":
    start_time = EndExhalation_tap[subject_id]['start'] - 0.1
    end_time = EndExhalation_tap[subject_id]['end'] + 0.1
elif breathing_hold == "EndInhalation":
    start_time = EndInhalation_tap[subject_id]['start'] - 0.1
    end_time = EndInhalation_tap[subject_id]['end'] + 0.1
else:
    start_time = 0
    end_time = 0
    print("Please define breathing_hold correctly")
    
    
start_frame = int(start_time*fps)
end_frame = int(end_time*fps)

# Reading start_frame
rawVideo.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
ret, first_frame = rawVideo.read()    
    
if not ret:
   print(f"Error reading first frame {first_frame}.")

# Select ROI from mouse 
window_name = "Select Region"
if scale_percent < 100:   
   resized_width = int(first_frame.shape[1]*scale_percent/100)
   resized_height = int(first_frame.shape[0]*scale_percent/100)
   dim = (resized_width, resized_height)
   resized_frames0 = cv2.resize(first_frame,dim,cv2.INTER_LINEAR)
   width_ratio = first_frame.shape[1] / resized_frames0.shape[1]
   height_ratio = first_frame.shape[0] / resized_frames0.shape[0]
   (xmin, ymin, boxw, boxh) = cv2.selectROI(window_name,resized_frames0)
   xmin = int(xmin*width_ratio)
   ymin = int(ymin*height_ratio)
   boxw = int(boxw*width_ratio)
   boxh = int(boxh*height_ratio)

else:
   (xmin, ymin, boxw, boxh) = cv2.selectROI(window_name,first_frame)
   
cv2.destroyWindow(window_name)
rect = [xmin, ymin, boxw, boxh]
print(rect)



# Output Directories
output_dir = os.path.join(path,subject_id,"Output", subject_id+'_'+breathing_hold, 'PPG')
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

opt_name = 'exp_ppg'
opt_exist_ok = False
output_path = Path(increment_path(Path(output_dir) / opt_name, exist_ok=opt_exist_ok))  # increment run
output_path.mkdir(parents=True, exist_ok=True)  # make dir

output_filename = str(output_path / filename[:-4]) + '.mp4'
out = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc(*'mp4v'),fps,(first_frame.shape[1],first_frame.shape[0]))

# for saving first frame when tap sound start
first_frame_fname = str(output_path / subject_id)+'_'+breathing_hold+'.jpg'
first_frame_draw = cv2.rectangle(first_frame, (int(round(xmin,0)), int(round(ymin,0))), (int(round(xmin+boxw,0)), int(round(ymin+boxh,0))), (255,255,0), 2)
cv2.imwrite(first_frame_fname, first_frame_draw)
out.write(first_frame_draw)


qr_xmin, qr_ymin, qr_w, qr_h = rect 

# Template image frame
It = cv2.cvtColor(first_frame,cv2.COLOR_RGB2GRAY)
template = It[int(qr_ymin):int(qr_ymin)+int(qr_h),int(qr_xmin):int(qr_xmin)+int(qr_w)]

prev_xmin, prev_ymin, prev_w, prev_h = rect
prev_xmax = prev_xmin + prev_w
prev_ymax = prev_ymin + prev_h

# Initialize list to store mean green values
blue_means = []     # channel 0
green_means = []    # channel 1
red_means = []      # channel 2

# Just for saving a value when the tap sound start
for k in range(start_frame):
    blue_means.append(0)
    green_means.append(0)
    red_means.append(0)

for i in range(start_frame,end_frame):
    if i==start_frame or i%100==0 or i==end_frame:
        print("Processing frame {}/{} of video {}".format(i,end_frame,filename)) 
        
    # read current frame
    ret, frame = rawVideo.read()    
    if not ret:
        print("FixMe: no Frame found. Out of index")        
        rawVideo.release()
        break 
    
    current_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)        
    It1 = current_frame[int(prev_ymin)-padding:int(prev_ymax)+padding, int(prev_xmin)-padding:int(prev_xmax)+padding]
    
    # Compute correlation map
    corrMap = cv2.matchTemplate(It1, template, cv2.TM_CCOEFF_NORMED)
    
    # Find the max and min value from the corrulation map
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corrMap)    
        
    # Transfer coordinates from the crop region to the original current frame
    prev_xmin = max_loc[0]+prev_xmin-padding
    prev_ymin = max_loc[1]+prev_ymin-padding    
    
    # for creating video
    frame_draw = frame.copy()
    
    # for ppg
    roi = frame[int(prev_ymin):int(prev_ymin)+int(qr_h), int(prev_xmin):int(prev_xmin)+int(qr_w), :]
    
    # Convert to float64 range 
    roi = roi.astype(np.float64)    
    
    blue_mean = np.mean(roi[:, :, 0])  
    green_mean = np.mean(roi[:, :, 1])    
    red_mean = np.mean(roi[:, :, 2])    

    blue_means.append(blue_mean)
    green_means.append(green_mean)
    red_means.append(red_mean)
    
    # video frame
    frame_draw = cv2.rectangle(frame_draw, (prev_xmin,prev_ymin), (prev_xmin+qr_w,prev_ymin+qr_h), (255,255,0), 2)
    out.write(frame_draw)

# Just for saving a value after the tap sound end
for j in range(end_frame, n_frame):
    blue_means.append(0)
    green_means.append(0)
    red_means.append(0)
    
# Stack the color channel data into a matrix
ppg_rgb = np.vstack((red_means, green_means, blue_means)).T

ppg_signal = bandpass_filter(green_means[start_frame:end_frame], lowcut=0.6, highcut=2.0, fs=fps) 


if ppg_signal is not None:
    print("PPG signal extracted successfully.")
    
    # Plot the PPG signal
    time_axis = np.arange(len(ppg_signal)) / fps
    plt.figure(figsize=(10, 2))
    plt.plot(time_axis, ppg_signal)
    plt.title('Extracted PPG Signal')
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.show()
else:
    print("Failed to extract PPG signal.")

# save ppg mat file
mat_file_name = 'ppg_'+subject_id+'_'+breathing_hold+'.mat'
scipy.io.savemat(str(output_path / mat_file_name), {'ppg_rgb': ppg_rgb})


out.release()
rawVideo.release()

print(subject_id)

