import cv2
import numpy as np

# normalize the image matrix to visualize
def normalization(img):

    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    output = (img.astype('float') - min_val) / (max_val - min_val) * 255

    return output


# read image
def read_image(path):
    rawimage = cv2.imread(path)
    # turn image into grayscale image
    rawimage = cv2.cvtColor(rawimage, cv2.COLOR_BGR2GRAY)
    height, width = rawimage.shape

    return rawimage, height, width


# do Gaussian filtering
def guassian_smoothing(rawimage, height, width):
    # initialize Gaussian operator
    gaussian = np.array([
        [1, 1, 2, 2, 2, 1, 1],
        [1, 2, 2, 4, 2, 2, 1],
        [2, 2, 4, 8, 4, 2, 2],
        [2, 4, 8, 16, 8, 4, 2],
        [2, 2, 4, 8, 4, 2, 2],
        [1, 2, 2, 4, 2, 2, 1],
        [1, 1, 2, 2, 2, 1, 1]
    ])

    # initial gussian filtering output (removing the border)
    gaussian_out = np.zeros([height, width])

    # do cross-correlation operation
    res = 0
    for i in range(3, height - 3):
        for j in range(3, width - 3):
            res = 0.0
            for m in range(7):
                for n in range(7):
                    res += rawimage[i + m - 3, j + n - 3] * gaussian[m, n]
            gaussian_out[i, j] = round(res / 140)

    # save the output image
    cv2.imwrite("./gaussian_output.bmp", gaussian_out)
    ga_height, ga_width = gaussian_out.shape

    return gaussian_out, ga_height, ga_width


# do sobel operation
def sobel_operation(gaussian_out, ga_height, ga_width):
    # horizontal sobel operator
    sobel_operator_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # vertical sobel operator
    sobel_operator_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    # initialize sobel-operation output
    sobel_xout = np.zeros([ga_height , ga_width], dtype=float)
    sobel_yout = np.zeros([ga_height , ga_width], dtype=float)

    # do cross-correlation operation
    resx = 0
    resy = 0
    for i in range(4, ga_height - 4):
        for j in range(4, ga_width - 4):
            resx = 0.0
            resy = 0.0
            for m in range(3):
                for n in range(3):
                    resx += gaussian_out[i + m - 1, j + n - 1] * sobel_operator_x[m, n]
                    resy += gaussian_out[i + m - 1, j + n - 1] * sobel_operator_y[m, n]
            sobel_xout[i, j] = resx
            sobel_yout[i, j] = resy

    # normalize the gradient
    # visualize it and save it
    sobel_xout_v = normalization(sobel_xout)
    sobel_yout_v = normalization(sobel_yout)

    cv2.imwrite("./sobel_xout.bmp", sobel_xout_v)
    cv2.imwrite("./sobel_yout.bmp", sobel_yout_v)

    return sobel_xout, sobel_yout

# calculate the magnitude
def magnitude(sobel_xout, sobel_yout):
    #so_height, so_width = sobel_yout.shape
    magnitude = np.sqrt(sobel_xout ** 2 + sobel_yout ** 2)
    # normorlize the magnitude
    # visulize it and save it
    magnitude_v = normalization(magnitude)
    cv2.imwrite("./magnitude.bmp", magnitude_v)

    return magnitude



def gradient_angle(sobel_xout, sobel_yout):
    # compute the angle of gradient (the output is in the range of [-pi,pi])
    angle = np.arctan2(sobel_yout, sobel_xout)

    return angle


# do non_maxima_suppression
def non_maxima_suppression(angle, magnitude):
    # quantize angle of the gradient to sector
    an_height, an_width = angle.shape
    for i in range(an_height):
        for j in range(an_width):
            if (- np.pi / 8) <= angle[i][j] < (np.pi / 8) or (7 * np.pi / 8) <= angle[i][j] < (9 * np.pi / 8) or (
                    -9 * np.pi / 8) <= angle[i][j] < (-7 * np.pi / 8):
                angle[i][j] = 0
            elif (np.pi / 8) <= angle[i][j] < (3 * np.pi / 8) or (-7 * np.pi / 8) <= angle[i][j] < (-5 * np.pi / 8):
                angle[i][j] = 1
            elif (3 * np.pi / 8) <= angle[i][j] < (5 * np.pi / 8) or (-5 * np.pi / 8) <= angle[i][j] < (-3 * np.pi / 8):
                angle[i][j] = 2
            elif (5 * np.pi / 8) <= angle[i][j] < (7 * np.pi / 8) or (-3 * np.pi / 8) <= angle[i][j] < (- np.pi / 8):
                angle[i][j] = 3

    # do non-maxima suppression (nms)
    nms_out = magnitude
    for i in range(5, an_height - 5):
        for j in range(5, an_width - 5):
            if angle[i][j] == 0:
                if magnitude[i][j] <= magnitude[i][j - 1] or magnitude[i][j] <= magnitude[i][j + 1]:
                    nms_out[i][j] = 0
            elif angle[i][j] == 1:
                if magnitude[i][j] <= magnitude[i - 1][j + 1] or magnitude[i][j] <= magnitude[i + 1][j - 1]:
                    nms_out[i][j] = 0
            elif angle[i][j] == 2:
                if magnitude[i][j] <= magnitude[i - 1][j] or magnitude[i][j] <= magnitude[i + 1][j]:
                    nms_out[i][j] = 0
            elif angle[i][j] == 3:
                if magnitude[i][j] <= magnitude[i - 1][j - 1] or magnitude[i][j] <= magnitude[i + 1][j + 1]:
                    nms_out[i][j] = 0

    # handle the edge cases in nms
    nms_height, nms_width = nms_out.shape
    nms_out[4][4], nms_out[-5][-5], nms_out[4][-5], nms_out[-5][4] = 0, 0, 0, 0
    for j in range(nms_width):
        if nms_out[4][j] != 0:
            nms_out[4][j] = 0
        if nms_out[-5][j] != 0:
            nms_out[-5][j] = 0
    for i in range(nms_height):
        if nms_out[i][4] != 2:
            nms_out[i][4] = 0
        if nms_out[i][-5] != 2:
            nms_out[i][-5] = 0

    # normorlize the magnitude after nms
    # visulize it and save it
    nms_out_v = normalization(nms_out)
    cv2.imwrite("./nms.bmp", nms_out_v)

    return nms_out, angle


# do double thresholding
def double_thresholding(nms_out, angle, t1):
    # initialize threshold 1 and threshold 2
    # set t1 method
    # method1: maxima * 0.1
    t1 = np.max(nms_out) * 0.1
    # method2: median * 0.66
    #t1 = np.median(nms_out) * 0.66
    # method3: mean * 0.66
    #t1 = np.mean(nms_out) * 0.66
    # method4: mannually
    #t1 = 12
    print t1
    t2 = 2 * t1
    nms_height, nms_width = nms_out.shape
    # initialize double-thresholding output
    threshold_out = np.ones([nms_height, nms_width])

    # assign the region according to t1,t2
    for i in range(nms_height):
        for j in range(nms_width):
            if nms_out[i][j] < t1:
                threshold_out[i][j] = 0
            elif nms_out[i][j] > t2:
                threshold_out[i][j] = 255
            else:
                continue

    # assign the region of those in range[t1,t2]
    for i in range(1, nms_height - 1):
        for j in range(1, nms_width - 1):
            if threshold_out[i][j] == 1:
                find_flag = 0
                for ni in range(-1, 2):
                    for nj in range(-1, 2):
                        if nms_out[i + ni][j + nj] > t2 and abs(angle[i + ni][j + nj] - angle[i][j]) <= np.pi / 4:
                            threshold_out[i][j] = 255
                            find_flag = 1
                            break
                    if find_flag == 1:
                        break
                if find_flag == 0:
                    threshold_out[i][j] = 0

    # visulize it and save it
    cv2.imwrite("./threshold.bmp", threshold_out)

    return threshold_out

if __name__ == '__main__':
    # enter image path
    path = '/Users/JesLee/Desktop/cv/project/canny_edge_detector/jl10919-project1/airplane256.bmp'
    #set threshold 1 manually or use one formula in the double_thresholding function
    t1 = 18
    # to present all data while printing a matrix
    np.set_printoptions(threshold=np.inf)
    #red image
    rawimage, height, width = read_image(path)
    #do gaussian smoothing
    gaussian_out, ga_height, ga_width = guassian_smoothing(rawimage, height, width)
    #do sobel operation
    sobel_xout, sobel_yout = sobel_operation(gaussian_out, ga_height, ga_width)
    #get magnitude
    magnitude = magnitude(sobel_xout, sobel_yout)
    #get angle
    angle = gradient_angle(sobel_xout, sobel_yout)
    #do non-maxima suppression
    nms_out, angle = non_maxima_suppression(angle, magnitude)
    #do double thresholding
    result = double_thresholding(nms_out, angle, t1)
