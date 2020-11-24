################################################ IMPORT ################################################################
# import the opencv library
import cv2
import time
import numpy


################################################ DECLARATIONS ##########################################################
# define a video capture object (webcam)
#vid = cv2.VideoCapture(0)

# with mobile phone
URL = 'http://192.168.10.102:6677/videoView'
vid = cv2.VideoCapture(URL)


# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# create array for getting average
lst = []

################################################ FUNCTIONS #############################################################
# get average of a list
def Average(lst):
    return sum(lst) / len(lst)

################################################ MAIN ##################################################################
