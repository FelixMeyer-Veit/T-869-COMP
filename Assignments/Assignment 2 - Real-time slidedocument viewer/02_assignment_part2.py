################################################ IMPORT ################################################################
import cv2
import time
import imutils
import random
from scipy.stats import linregress

import numpy as np
from matplotlib import pyplot as plt
import scipy # use numpy if scipy unavailable
import scipy.linalg # use numpy if scipy unavailable

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
    #cv2.putText(frame, fps_avg, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    return prev_frame_time

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

################################################ DECLARATIONS ##########################################################
# define a video capture object (webcam)
video = cv2.VideoCapture(0)
video.open("http://192.168.10.104:8080/video") # use android phone with app IP webcam

# to save all the clicked images
img_counter = 0
scn_counter = 0

##### for computing FPS
# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0
# create array for getting average
lst = []
####

################################################ MAIN ##################################################################

while (True):
    ret, frame = video.read()
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    if key == ord('c'):
        img_name = "Assignment2_4lines_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    img = frame
    image = img
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = cv2.Canny(gray, 75, 200)
    # edged = cv2.Canny(image, 50, 125)

    # show the original image and the edge detected image
    cv2.imshow("Original frame", image)
    #cv2.imshow("Gray+GaussianBlur+Canny", edged)

    # Find the contours and sort them
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # show the contour (outline) of the piece of paper
    # print("STEP 2: Find contours of paper")
    if (len(approx) == 4):
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        warpedShow = four_point_transform(orig, (screenCnt.reshape(4, 2) * ratio))
        cv2.imshow("Original frame with outlined object", image)
        cv2.imshow("Warped frame", warpedShow)
    if key == ord('s'):  # Scan
        scn_counter += 1
        # apply the four point transform to obtain a top-down
        # view of the original image
        warped = four_point_transform(orig, (screenCnt.reshape(4, 2) * ratio))
        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        ####warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # show the original and scanned images
        scan_name = 'Scanned_outlined_object{}.png'.format(scn_counter)
        outlined_name = 'Outlined_frame_{}.png'.format(scn_counter)
        cv2.imwrite(outlined_name, image)
        cv2.imwrite(scan_name, warped)
        print("Outlined frame saved as: {}".format(outlined_name))
        print("scanned object saved as: {}".format(scan_name))
        print(" ")

# release the capture
video.release()
cv2.destroyAllWindows()



while (True):

    # Capture the video frame by frame
    ret, frame = vid.read()
    ################################################

    frame = frame[120:360, 160:480, :]  # reduce frame-size

    # get edges and their points
    edges = cv2.Canny(frame, 100, 200)
    Y, X, N_points = get_edge_points(edges)

    # get edge with the most inliers
    RANSAC(Y, X, N_points, frame)

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get fps and store a new prev_frame_time
    prev_frame_time = get_fps(frame, prev_frame_time)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    '''
    # press 'c' to capture image
    elif cv2.waitKey(10) & 0xFF == ord('c'):
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        frame_canny = get_canny()
        ransac(frame_canny)
    '''

# Driver Code
average = Average(lst)

# Printing average of the list
print("Average processing time =", round(average, 2))

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

################################################ OLD FUNCTIONS #########################################################
