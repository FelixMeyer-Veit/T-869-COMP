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

def getSpot(dest, frame,g, b, color):
    # convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g_gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    # apply a Gaussian blur to the image then find the brightest region
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    g_gray = cv2.GaussianBlur(g_gray, (1, 1), 0)
    b_gray = cv2.GaussianBlur(b_gray, (1, 1), 0)

    # get reddest spot
    if color == 1:
        gray = gray - g_gray - b_gray

    # perform a naive attempt to find the (x, y) coordinates of
    # the area of the image with the largest intensity value
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # puting the Location on the frame
    if color == 1:
        cv2.circle(dest, maxLoc, 5, (0, 0, 255), 2)
        maxLoc = str(maxLoc)
        cv2.putText(dest, maxLoc, (400, 450), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    elif color == 0:
        cv2.circle(dest, maxLoc, 5, (255, 0, 0), 2)
        maxLoc = str(maxLoc)
        cv2.putText(dest, maxLoc, (7, 450), font, 1, (255, 0, 0), 1, cv2.LINE_AA)

'''
def getSpot(dest, frame, color):
    # convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply a Gaussian blur to the image then find the brightest region
    gray = cv2.GaussianBlur(gray, (1, 1), 0)

    # perform a naive attempt to find the (x, y) coordinates of
    # the area of the image with the largest intensity value
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # puting the Location on the frame
    if color == 1:
        cv2.circle(dest, maxLoc, 5, (0, 0, 255), 2)
        maxLoc = str(maxLoc)
        cv2.putText(dest, maxLoc, (400, 450), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    elif color == 0:
        cv2.circle(dest, maxLoc, 5, (255, 0, 0), 2)
        maxLoc = str(maxLoc)
        cv2.putText(dest, maxLoc, (7, 450), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
'''

def getSpot2(dest, frame, g, b, color):
    # convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g_gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    # apply a Gaussian blur to the image then find the brightest region
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    g_gray = cv2.GaussianBlur(g_gray, (1, 1), 0)
    b_gray = cv2.GaussianBlur(b_gray, (1, 1), 0)

    # get reddest spot
    if color == 1:
        gray = gray - g_gray - b_gray

    # perform a naive attempt to find the (x, y) coordinates of
    # the area of the image with the largest intensity value

    rows, cols = gray.shape[:2]
    max = 0
    maxLoc = (0,0)
    for x in range(cols):
        for y in range(rows):
            temp = frame[y, x]
            if temp[2] > max:
                max = temp[2]
                maxLoc = (y, x)

    if color == 0:
        cv2.circle(dest, maxLoc, 5, (255, 0, 0), 2)
        maxLoc = str(maxLoc)
        cv2.putText(dest, maxLoc, (7, 450), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
    elif color == 1:
        cv2.circle(dest, maxLoc, 5, (0, 0, 255), 2)
        maxLoc = str(maxLoc)
        cv2.putText(dest, maxLoc, (400, 450), font, 1, (0, 0, 255), 1, cv2.LINE_AA)


'''
# get reddest spot with exact subtraction/calculation --> takes more time

def getSpot2(dest, frame, color):
    # convert image to grayscale
    if color == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # apply a Gaussian blur to the image then find the brightest region
        gray = cv2.GaussianBlur(gray, (1, 1), 0)

        # perform a naive attempt to find the (x, y) coordinates of
        # the area of the image with the largest intensity value

        rows, cols = gray.shape[:2]
        max = 0
        maxLoc = (0,0)
        for x in range(cols):
            for y in range(rows):
                temp = frame[y, x]
                if temp[2] > max:
                    max = temp[2]
                    maxLoc = (y, x)

        cv2.circle(dest, maxLoc, 5, (255, 0, 0), 2)
        maxLoc = str(maxLoc)
        cv2.putText(dest, maxLoc, (7, 450), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
    elif color == 1:
        # perform a naive attempt to find the (x, y) coordinates of
        # the area of the image with the largest intensity value

        rows, cols = dest.shape[:2]
        max = 0
        maxLoc = (0, 0)
        for x in range(cols):
            for y in range(rows):
                temp = dest[y, x]
                sum = int(temp[2]) - int(temp[1]) - int(temp[0])
                if sum > max:
                    max = sum
                    maxLoc = (y, x)

        cv2.circle(dest, maxLoc, 5, (0, 0, 255), 2)
        maxLoc = str(maxLoc)
        cv2.putText(dest, maxLoc, (400, 450), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
'''

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

################################################ MAIN ##################################################################
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get fps and store a new prev_frame_time
    prev_frame_time = get_fps(frame, prev_frame_time)

    # get the red channel
    r = frame.copy()
    # set blue and green channels to 0
    r[:, :, 0] = 0
    r[:, :, 1] = 0

    b = frame.copy()
    # set green and red channels to 0
    b[:, :, 1] = 0
    b[:, :, 2] = 0

    g = frame.copy()
    # set blue and red channels to 0
    g[:, :, 0] = 0
    g[:, :, 2] = 0

    getSpot2(frame, r, g, b, 1) # 1 = get reddest spot
    getSpot2(frame, frame, g, b, 0)  # 0 = get brightest spot

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Driver Code
average = Average(lst)

# Printing average of the list
print("Average processing time =", round(average, 2))

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
