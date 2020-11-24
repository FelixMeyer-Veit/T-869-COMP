################################################ IMPORT ################################################################
import cv2
import time
import numpy
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
def ransac(data, model, n, k, t, d, debug=False, return_all=False):

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