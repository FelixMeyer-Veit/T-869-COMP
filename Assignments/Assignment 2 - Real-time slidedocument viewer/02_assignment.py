################################################ IMPORT ################################################################
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

################################################ FUNCTIONS #############################################################
# get average of a list
def Average(lst):
    return sum(lst) / len(lst)

def get_fps(frame, prev_frame_time):
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # add current fps at the end of array
    lst.append(fps)

    # get the average of processing time
    average = int(Average(lst))

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps_avg = str(average)

    # puting the FPS count on the frame
    cv2.putText(frame, fps_avg, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    return prev_frame_time

def create_edge():
    img = cv2.imread("{}".format(img_name), 0)
    edges = cv2.Canny(img, 100, 200)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

################################################ DECLARATIONS ##########################################################
# define a video capture object (webcam)
vid = cv2.VideoCapture(0)

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

# create array for getting average
lst = []

img_counter = 0

################################################ MAIN ##################################################################
while (True):

    # Capture the video frame by frame
    ret, frame = vid.read()

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get fps and store a new prev_frame_time
    prev_frame_time = get_fps(frame, prev_frame_time)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # press 'c' to capture image
    if cv2.waitKey(1) & 0xFF == ord('c'):
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        create_edge()


# Driver Code
average = Average(lst)

# Printing average of the list
print("Average processing time =", round(average, 2))

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()