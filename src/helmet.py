from os import environ
def suppress_qt_warnings():
        environ["QT_DEVICE_PIXEL_RATIO"] = "0"
        environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        environ["QT_SCREEN_SCALE_FACTORS"] = "1"
        environ["QT_SCALE_FACTOR"] = "1"

import shutil 	
from time import sleep
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from glob import glob

frame_count = 0              

confThreshold = 0.5  
nmsThreshold = 0.4   
inpWidth = 416       
inpHeight = 416      

classesFile = "obj.names";
classes = None
with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')


modelConfiguration = "yolov3-obj.cfg";
modelWeights = "yolov3-obj_2400.weights";
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    
def getOutputsNames(net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    
def drawPred(classId, conf, left, top, right, bottom):

        global frame_count

        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        label = '%.2f' % conf
        
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])

        label_name,label_conf = label.split(':')    
        if label_name == 'helmet':
                                                
            cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            frame_count+=1
        elif label_name == 'person':
                                                
            cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            frame_count+=1

    	

        
        if(frame_count> 0):
            return frame_count




    
def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        global frame_count_out
        frame_count_out=0
        classIds = []
        confidences = []
        boxes = []
        classIds = []               
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        count_person=0 
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
                   
            frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
            

            my_class='helmet'                   
            unknown_class = classes[classId]

            if my_class == unknown_class:
                count_person += 1
        if count_person >= 1:
            path = 'test_out/'
            frame_name=os.path.basename(fn)             
            cv.imwrite(str(path)+frame_name, frame)     
            cv.imshow('img',frame)
            cv.waitKey(800)



        
    
suppress_qt_warnings()
winName = 'Deep learning object detection in OpenCV'
    



for fn in glob('.images/*.jpg'):
		frame = cv.imread(fn)
		frame = cv.resize(frame, (960, 540))
		frame_count =0

        
		blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
		net.setInput(blob)

        
		outs = net.forward(getOutputsNames(net))

        
		postprocess(frame, outs)

		t, _ = net.getPerfProfile()
        
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        
		cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
       

directory = '.images'
directory1 = 'test_out'
for filename in glob('.images/*.jpg'):
		if (os.path.basename(filename).endswith(".jpg") or os.path.basename(filename).endswith(".png")) and os.path.basename(filename) not in os.listdir(directory1):
			frame = cv.imread(".images/" + os.path.basename(filename))
			imS = cv.resize(frame, (960, 540))
			cv.imshow('img',imS)
			cv.waitKey(800)
			shutil.copy(filename,"violation")
		else:
			continue