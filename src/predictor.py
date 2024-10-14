# import the necessary packages
import numpy as np
import pickle
import cv2
from skimage.feature import hog,local_binary_pattern
from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
from time import sleep

print("Building model")
recognizer = pickle.loads(open("classifier.pickle", "rb").read())

le = pickle.loads(open("le.pickle", "rb").read())
print("Model loaded")

light_timer = {
				15:range(0,30),
				20:range(30,50),
				30:range(50,70),
				45:range(70,90),
				60:range(90,100)
			}

def img_read(cap, ic, jc, density):
        eps=1e-7
        numPoints = 24
        radius = 8
        _,img = cap.read()
        img = cv2.resize(img,(800,600))
        roi = img[80:435,270:670]
        col = roi.copy()
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        h,w = roi.shape
        
        ic+=1
        if(ic <= 150):
            #cv2.imshow('grayed out',roi)
            #cv2.waitKey(1)
            return (-1, ic, jc, density)
        
        if(ic%8 == 0):
            positive_count = 0
            negative_count = 0
            for i in range(44,h,44):
                for j in range(44,w,44):
                    box = roi[i-44:i,j-44:j]
                    lbp = local_binary_pattern(box, numPoints, radius, method="uniform")
                    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))

                    hist = hist.astype("float")
                    hist /= (hist.sum() + eps)
     
                    lbp_embedding = hist
                    hog_embedding = hog(box, orientations=8, pixels_per_cell=(3, 3), cells_per_block=(1, 1), visualize=False, multichannel=False)
                    embedding = np.append(hog_embedding.ravel(),lbp_embedding)

                    prediction = recognizer.predict(embedding.reshape(1, -1))

                    if(prediction == 1):
                        cv2.rectangle(col,(j,i),(j-39,i-39),(0,0,255),1)
                        positive_count += 1
                    else:
                        cv2.rectangle(col,(j,i),(j-39,i-39),(0,255,0),1)
                        negative_count += 1
                    jc+=1
            density.append(positive_count/(positive_count+negative_count))
            train, test = density[0:-1], density[-1:]
            if(len(density) > 10):
                model = ARIMA(density, order=(5, 1, 1))
                model_fit = model.fit()
                output = model_fit.forecast()
                predicted_value = output[0]
                return predicted_value, ic, jc, density
            else:
                return positive_count/(positive_count+negative_count), ic, jc, density
        return -1, ic, jc ,density
            #cv2.imshow('grids',col)
            #cv2.waitKey(1)
    #cv2.destroyAllWindows()

lane1 = cv2.VideoCapture("lane1.mp4")
lane2 = cv2.VideoCapture("lane2.mp4")
lane3 = cv2.VideoCapture("lane4.mp4")
lane4 = cv2.VideoCapture("lane4.mp4")
ic1, jc1 = 0, 0
ic2, jc2 = 0, 0
ic3, jc3 = 0, 0
ic4, jc4 = 0, 0
density1 = []
density2 = []
density3 = []
density4 = []
print("Starting...\n\n")
while(True):
    #print(img_read(lane1, ic1, jc1, density1))
    a, ic1, jc1, density1 = img_read(lane1, ic1, jc1, density1)
    b, ic2, jc2, density2 = img_read(lane2, ic2, jc2, density2)
    c, ic3, jc3, density3 = img_read(lane3, ic3, jc3, density3)
    d, ic4, jc4, density4 = img_read(lane4, ic4, jc4, density4)
    print("Densities: ",a, b, c, d, "\n")
    if(a==-1 or b==-1 or c==-1 or d==-1):
        continue
    if(a >= max(b, c, d)):
        print("Lane: ", "NorthBound")
        print("Status: ", "Green")
        print("Density: ", a*100)
        for i in light_timer:
            if(int(a*100) in light_timer[i]):
                timer = i
        print("Time: ", timer, "\n")
        print("Lane: ", "EastBound")
        print("Status: ", "Red")
        print("Density: ", b*100)
        print("Time: ", "N/A", "\n")
        print("Lane: ", "SouthBound")
        print("Status: ", "Red")
        print("Density: ", c*100)
        print("Time: ", "N/A", "\n")
        print("Lane: ", "WestBound")
        print("Status: ", "Red")
        print("Density: ", d*100)
        print("Time: ", "N/A", "\n")
        sleep(timer)
        continue
    if(b >= max(a, c, d)):
        print("Lane: ", "NorthBound")
        print("Status: ", "Red")
        print("Density: ", a*100)
        print("Time: ", "N/A", "\n")
        print("Lane: ", "EastBound")
        print("Status: ", "Green")
        print("Density: ", b*100)
        for i in light_timer:
            if(int(b*100) in light_timer[i]):
                timer = i
        print("Time: ", timer, "\n")
        print("Lane: ", "SouthBound")
        print("Status: ", "Red")
        print("Density: ", c*100)
        print("Time: ", "N/A", "\n")
        print("Lane: ", "WestBound")
        print("Status: ", "Red")
        print("Density: ", d*100)
        print("Time: ", "N/A", "\n")
        sleep(timer)
        continue
    if(c >= max(b, a, d)):
        print("Lane: ", "NorthBound")
        print("Status: ", "Red")
        print("Density: ", a*100)
        print("Time: ", "N/A", "\n")
        print("Lane: ", "EastBound")
        print("Status: ", "Red")
        print("Density: ", b*100)
        print("Time: ", "N/A", "\n")
        print("Lane: ", "SouthBound")
        print("Status: ", "Green")
        print("Density: ", c*100)
        for i in light_timer:
            if(int(c*100) in light_timer[i]):
                timer = i
        print("Time: ", timer, "\n")
        print("Lane: ", "WestBound")
        print("Status: ", "Red")
        print("Density: ", d*100)
        print("Time: ", "N/A", "\n")
        sleep(timer)
        continue
    if(d >= max(b, c, a)):
        print("Lane: ", "NorthBound")
        print("Status: ", "Red")
        print("Density: ", a*100)
        print("Time: ", "N/A", "\n")
        print("Lane: ", "EastBound")
        print("Status: ", "Red")
        print("Density: ", b*100)
        print("Time: ", "N/A", "\n")
        print("Lane: ", "SouthBound")
        print("Status: ", "Red")
        print("Density: ", c*100)
        print("Time: ", "N/A", "\n")
        print("Lane: ", "WestBound")
        print("Status: ", "Green")
        print("Density: ", d*100)
        for i in light_timer:
            if(int(d*100) in light_timer[i]):
                timer = i
        print("Time: ", timer, "\n")
        sleep(timer)
        continue
