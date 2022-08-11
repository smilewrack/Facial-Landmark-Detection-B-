import numpy as np
import imutils
from imutils import face_utils
import dlib
import cv2
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import math

import sys
sys.path.append('.')
from utils import plot_landmarks, frontalize_landmarks, get_landmark_array

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
frontalization_weights = np.load('frontalization_weights.npy')

'''
Input: captured video
Output: numpy array that has x,y coordinates of 68 face keypoints
(array shape: [number of frames,68,2])
'''
def KeyPoint(video, detector, predictor):
    repeat = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    arr = []
    
    for frame in range(repeat):
        ret, image = video.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            arr.append(shape)
            
    video.release()
    arr = np.array(arr)
    return arr

'''
calculating distance
Input: [x,y]
Output: float
'''
def dist(a,b):
    return ((a[0]-b[0])*(a[0]-b[0])+(a[1]-b[1])*(a[1]-b[1])) ** 0.5

'''
return average coordinates of the list of dots
Input: index of dots(numbers)
Output: average coordinate(x coordinate, y coordinate)
'''
def avglist(lst, arr):
    res = np.zeros(2)
    for i in lst:
        res += arr[i]
    return res/len(lst)

def cal_dist(d1, d2, d3):
    area = abs((d1[0]-d3[0]) * (d2[1]-d3[1]) - (d1[1]-d3[1]) * (d2[0] - d3[0]))
    return area/dist(d1,d2)

'''
catches miscoordinated dots and fixes them
Input: numpy array that has x,y coordinates of 68 face keypoints
(array shape: [number of frames,68,2])
Output: numpy array that has x,y coordinates of 68 face keypoints
(array shape: [number of frames,68,2])
'''
def correction(array):
    arr = np.array(array)
    res = np.zeros(arr.shape)
    for i in range(len(arr)):
        res[i] += frontalize_landmarks(arr[i], frontalization_weights)
    
    tmp = [dist(avglist([50,51,52,61,62,63],x),avglist([56,57,58,65,66,67],x)) for x in res]
    tmp2 = [[dist(x[i],x[16-i]) for x in res] for i in range(8)]
    tmp, tmp2 = np.array(tmp), np.array(tmp2)
    
    for i in range(8):
        line_fitter = LinearRegression()
        line_fitter.fit(tmp.reshape(-1,1), tmp2[i])
        predicted = line_fitter.predict(tmp.reshape(-1,1))
        rmse = math.sqrt(mean_squared_error(tmp2[i],predicted))
        standard = (tmp2[i] - predicted) / rmse
        for j in range(len(standard)):
            if abs(standard[j]) < 2:
                continue
            oklist1 = []
            oklist2 = []
            for x in range(max(0,j-3),min(len(standard),j+4)):
                if abs(standard[x]) < 2:
                    oklist1.append(arr[x][i])
                    oklist2.append(arr[x][16-i])
            oklist1 = np.array(oklist1)
            oklist2 = np.array(oklist2)
            if len(oklist1):
                arr[j][i] = np.mean(oklist1,axis=0)
                arr[j][16-i] = np.mean(oklist2,axis=0)
    
    return arr       

'''
averaging keypoints with adjacent 5 frames by giving weights
Input: numpy array that has x,y coordinates of 68 face keypoints
(array shape: [number of frames,68,2])
Output: numpy array that has x,y coordinates of 68 face keypoints
(array shape: [number of frames,68,2])
'''
def avg5(arr):
    res = np.zeros((arr.shape[0]+4,68,2))
    for i in range(5):
        res += np.concatenate([np.zeros((i,68,2)),arr,np.zeros((4-i,68,2))])
    res /= 5
    res = np.concatenate([arr[:2],res[4:-4],arr[-2:]])
    res = [np.vstack([res[i][:48],arr[i][48:]]) for i in range(len(arr))]
    res = np.array(res)
    return res

'''
shows the keypoint within the video
Input: captured video, list of keyframe datas, list of dot colors
Output: video with the keypoints of (multiple) model(s)
'''
def showdot(video, datalist, colorlist):
    repeat = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) 
    for i in range(repeat):
        ret, image = video.read()
        for data, color in zip(datalist, colorlist):
            for (x, y) in data[i]:
                image = cv2.circle(image, (int(x), int(y)), 1, color, -1)
        cv2.imshow('frame',image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    video.release()
    cv2.destroyAllWindows()

'''
rotate the coordinates by deg
Input: x coordinates, y coordinates, degree
Output: rotated x coordinates, rotated y coordinates
'''
def rotate(x, y, deg):
    x1 = x * math.cos(deg) - y * math.sin(deg)
    y1 = x * math.sin(deg) + y * math.cos(deg)
    return x1, y1

'''
fit the dots into polynomial function, and calculates mse
Input: x coordinates, y coordinates
Output: MSE
'''
def polyreg(x, y):
    poly_features = PolynomialFeatures(degree = 2, include_bias=False)
    x_poly = poly_features.fit_transform(x)

    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)

    x_new_poly = poly_features.transform(x)
    y_new = lin_reg.predict(x_new_poly)
    mse = mean_squared_error(y, y_new)

    return mse

'''
rotates the dot to find the function that minimalizes mse
Input: x coordinates, y coordinates
Output: MSE
'''
def bestreg(x, y):
    val = math.inf
    for deg in range(0,180,45):
        x1, y1 = rotate(x, y, math.radians(deg))
        mse = polyreg(x1, y1)
        val = min(mse,val)
    return val

'''
calculates rmse of the keypoints, which is used to check stability
Input: numpy array that has x,y coordinates of 68 face keypoints
(array shape: [number of frames,68,2])
Output: RMSE
'''
def calrmse(data):
    arr = np.transpose(data, (1,2,0))
    msesum = 0
    for dot in arr:
        for i in range(0, len(data)-1, 7):
            x = dot[0][i:i+10].reshape(-1,1)
            y = dot[1][i:i+10].reshape(-1,1)
            msesum += bestreg(x, y)
            
    return math.sqrt(msesum/len(data)*7)

'''
finds the keypoints of video frames and saves 2 files;
before correction and after correction.
If there are frames that are failed to find keypoints
or found more than 2 keypoints in one frame, files 
will not be saved since correction doesn't work.
Input: video's path, before file's name(or path), after file's name(or path)
'''
def save_file(video_path,before_file,after_file):
    capture = cv2.VideoCapture(video_path)
    before = KeyPoint(video, detector, predictor)
    if len(before) != int(capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        print('keypoints were not found properly')
        return
    np.save(before_file,before)
    after = correction(before)
    after = avg5(after)
    np.save(after_file,after)

'''
show video with keypoints, comparing before and after
dot color -> before: red, after: white
'''
def compare_keypoint(video_path,before_path,after_path):
    capture = cv2.VideoCapture(video_path)
    datalist = [np.load(before_path),np.load(after_path)]
    showdot(capture, datalist, [(0,0,255),(255,255,255)])