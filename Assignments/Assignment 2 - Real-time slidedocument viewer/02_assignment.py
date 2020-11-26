################################################ IMPORT ################################################################
import cv2
import time
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

def get_edge_points(edges):
    edge_points = np.where(edges == 255)
    Y = edge_points[0]  # 480 (Vertical)
    X = edge_points[1]  # 640 (Horizontal)
    N_points = len(Y)
    return Y, X, N_points

def RANSAC(Y, X, N_points, frame):
    # Declarations
    np.random.seed(1)
    slope_is_infinity = False

    # parameter-optimization
    # n_iter = int(np.log(1-0.95)/np.log(1-(1-0.4)*(1-0.4)))
    n_iter = 50
    Delta_Ransac = 1
    for i in range(n_iter):
        # get 2 random samples
        Random_points = np.random.randint(0, N_points, 2)
        x_rand_0 = X[Random_points[0]]
        y_rand_0 = Y[Random_points[0]]
        x_rand_1 = X[Random_points[1]]
        y_rand_1 = Y[Random_points[1]]

        # calculate slope and intercept when having 2 random points
        DeltaX = x_rand_0 - x_rand_1
        DeltaY = y_rand_0 - y_rand_1
        if DeltaX == 0:
            slope = 0  # should be a high number (infinity), but received better results when having 0
            slope_is_infinity = True
        else:
            slope = DeltaY / DeltaX
        intercept = y_rand_0 - slope * x_rand_0

        # get boundaries for inliers
        angle = np.arctan(np.deg2rad(slope))
        temp_length = Delta_Ransac / np.cos(angle)
        Intercept_Lower = intercept + temp_length
        Intercept_Upper = intercept - temp_length

        if slope_is_infinity == True:  # infinity slope
            inliers = np.where(((X > x_rand_0 - 5) & (X < x_rand_0 + 5)), 1, 0)  # if it is in range (x_rand_0=x_rand_1 and it is very steep --> look for close X around given x), then add 1; else 0 to the list
        else:
            inliers = np.where((Y > (slope * X + Intercept_Upper)) & (Y < (slope * X + Intercept_Lower)), 1,
                               0)  # if it is in range, then add 1; else 0 to the list
        sum_inliers = sum(inliers)
        if i == 0:  # first iteration
            most_inliers = sum_inliers
            best_intercept = intercept
            best_slope = slope
            best_x_0 = x_rand_0
            best_x_1 = x_rand_1
            #best_y_0 = y_rand_0
            #best_y_1 = y_rand_1
            if slope_is_infinity == True:
                line_is_vertical = True
                slope_is_infinity = False
            else:
                line_is_vertical = False

        else:  # all the other iterations
            if sum_inliers > most_inliers:
                most_inliers = sum_inliers
                best_intercept = intercept
                best_slope = slope
                best_x_0 = x_rand_0
                best_x_1 = x_rand_1
                #best_y_0 = y_rand_0
                #best_y_1 = y_rand_1
                if slope_is_infinity == True:
                    line_is_vertical = True
                    slope_is_infinity = False
                else:
                    line_is_vertical = False
            else:
                slope_is_infinity = False

    if line_is_vertical == True:  # vertical lines
        cv2.line(frame, (best_x_0, 0), (best_x_1, 500), (0, 255, 0), 2)
    else:  # horizontal lines
        cv2.line(frame, (0, int((best_slope * -0 + best_intercept))), (400, int((best_slope * 400 + best_intercept))),
                 (255, 0, 0), 2)


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
'''
def resize(frame):
    print('Original Dimensions : ', frame.shape)

    scale_percent = 40  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(frame, dim)

    print('Resized Dimensions : ', resized.shape)
    return resized

def get_canny():
    img = cv2.imread("{}".format(img_name), 0)
    edges = cv2.Canny(img, 50, 100)
    cv2.imshow('Edges', edges)

    #plt.subplot(121), plt.imshow(img, cmap='gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122), plt.imshow(edges, cmap='gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.show()

    return edges
'''
'''
def get_line(x1,y1,x2,y2):
    slope, intercept, r_value, p_value, std_err = linregress([x1, x2], [y1, y2])
    print(slope, intercept)
    return slope, intercept

def ransac(frame):
    # resize canny-image
    resized = resize(frame)
    rows, cols = resized.shape[:2]
    cv2.imshow("Edge-resized", resized)

    # get all edge coordinates
    min = 200
    edge_coordinates = []
    for x in range(cols):
        for y in range(rows):
            value = frame[y, x]
            print(value)
            if value > min:
                edge_coordinates.append((y, x))

    print("Edge Coordinates list: " , edge_coordinates)

    iterations = 10
    count_max = 0
    for iter in range(iterations):
        # pick 2 random points from edge coordinates and remove them
        if len(edge_coordinates) > 0:
            max = len(edge_coordinates) - 1
            print("len of edgecoordinates1 after iter", iter, max)
            rand1 = random.randint(0, max)
            x1, y1 = edge_coordinates[rand1]
            edge_coordinates.remove((x1, y1))
            max = len(edge_coordinates) - 1
            print("len of edgecoordinates2 after iter", iter, max)
            rand2 = random.randint(0, max)
            x2, y2 = edge_coordinates[rand2]
            edge_coordinates.remove((x2, y2))

            # get slope and intercept of created linear function for random points
            m, b = get_line(x1, y1, x2, y2)

            # find inliers
            delta = 2
            inlier_count = 0
            for coordinate in edge_coordinates:
                x = coordinate[0]
                y = coordinate[1]
                y1 = m*x + b + delta
                y2 = m*x + b - delta
                if y < y1 and y > y2:
                    inlier_count += 1
                    edge_coordinates.remove((x, y))

            if(inlier_count > count_max):
                count_max = inlier_count
                print("inlier counts", count_max)
                best_fit = (m, b, int(x1), int(y1), int(x2), int(y2))
        else:
            iter = iterations - 1
        iter += 1

    drawline(best_fit, resized)

def drawline(bestfit, resized):
    m = bestfit[0]
    b = bestfit[1]
    x = resized.shape[1]-1
    y = m * x + b

    #start_point = (0, int(b))
    #end_point = (x, int(y))
    start_point = (bestfit[2], bestfit[3])
    end_point = (bestfit[4], bestfit[5])
    #start_point = (0, 0)
    #end_point = (480, 640)

    print("start_point: ", start_point, "and end_point: ", end_point)
    #print("start_point_bestfit: ", bestfit[2], "and end_point: ", bestfit[3])

    # Green color in BGR
    color = (255, 255, 0)
    # Line thickness of 4 px
    thickness = 4

    # Using cv2.line() method
    # Draw a green line with thickness of 9 px
    resized = cv2.line(resized, start_point, end_point, color, thickness)
    cv2.imshow('result', resized)



def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """fit model parameters to data using the RANSAC algorithm

   This implementation written from pseudocode found at
   http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
   {{{
   Given:
       data - a set of observed data points
       model - a model that can be fitted to data points
       n - the minimum number of data values required to fit the model
       k - the maximum number of iterations allowed in the algorithm
       t - a threshold value for determining when a data point fits a model
       d - the number of close data values required to assert that a model fits well to data
   Return:
       bestfit - model parameters which best fit the data (or nil if no good model is found)
   iterations = 0
   bestfit = nil
   besterr = something really large
   while iterations < k {
       maybeinliers = n randomly selected values from data
       maybemodel = model parameters fitted to maybeinliers
       alsoinliers = empty set
       for every point in data not in maybeinliers {
           if point fits maybemodel with an error smaller than t
                add point to alsoinliers
       }
       if the number of elements in alsoinliers is > d {
           % this implies that we may have found a good model
           % now test how good it is
           bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
           thiserr = a measure of how well model fits these points
           if thiserr < besterr {
               bestfit = bettermodel
               besterr = thiserr
           }
       }
       increment iterations
   }
   return bestfit
   }}}
   """

    iterations = 0
    bestfit = None
    besterr = numpy.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = scipy.random_partition(n, data.shape[0])
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < t]  # select indices of rows with accepted points
        alsoinliers = data[also_idxs, :]
        if debug:
            print
            'test_err.min()', test_err.min()
            print
            'test_err.max()', test_err.max()
            print
            'numpy.mean(test_err)', numpy.mean(test_err)
            print
            'iteration %d:len(alsoinliers) = %d' % (
                iterations, len(alsoinliers))
        if len(alsoinliers) > d:
            betterdata = numpy.concatenate((maybeinliers, alsoinliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = numpy.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = numpy.concatenate((maybe_idxs, also_idxs))
        iterations += 1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit
'''