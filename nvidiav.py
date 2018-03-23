import cv2
import numpy as np
import sys
import logging
import time
import dbus
from networktables import NetworkTables

#time.sleep(20)
def check_point(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDBLCLK:
		pixel = hsv[y,x]
		hold = np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
		convert = np.array([[hsv[x,y]]])
		print x
		print y
		print (hold)

vid = cv2.VideoCapture(0)
#vid.set(4,1280)
#vid.set(5,720)
#vid.set(10,.05)

cv2.namedWindow('GaussianBlur')
cv2.setMouseCallback('GaussianBlur',check_point)
logging.basicConfig(level=logging.DEBUG)
minpixels=200
minpixels_cube=800


#while not NetworkTables.isConnected():
print NetworkTables.initialize(server = '192.168.0.110')
time.sleep(1)
statustable = NetworkTables.getTable("status")
statustable.putBoolean('booted', True)
vtargetobj = NetworkTables.getTable("vtargetobj")
cubetarget = NetworkTables.getTable("cubetarget")

while(True):
    if statustable.getEntry('powerstatus').getBoolean(True)== False:
        break
    ret, frame = vid.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #covert to hsv space
    gBlurImg = cv2.GaussianBlur(hsv, (9,9), 1.7) #gaussian blur for noise reduction

    #cv2.imshow('orig',hsv)
    cv2.imshow("GaussianBlur", gBlurImg)

    lower_green = np.array([60,60,150])
    upper_green = np.array([110,110,255])

    mask = cv2.inRange(gBlurImg, lower_green, upper_green)
    res = cv2.bitwise_and(frame,frame,mask=mask)

    imgray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    targetlistx=[]
    targetlisty=[]
    targetlistw=[]
    targetlisth=[]
    if len(contours) !=0:
	i=0
	while (i<len(contours)):
		if(cv2.contourArea(contours[i])>minpixels):
			comp=hierarchy[0,i,3]
			if (comp==-1):
				x,y,w,h =cv2.boundingRect(contours[i])
				targetlistx.append(x)
				targetlisty.append(y)
				targetlistw.append(w)
				targetlisth.append(h)
				cv2.rectangle(res, (x,y), (x+w, y+h), (0,0,255),2)
		i=i+1
    
    cv2.imshow('fff',res)
    vtargetobj.putNumber('objcount', len(targetlistx))
    vtargetobj.putNumberArray('vtargetobjx',targetlistx)
    vtargetobj.putNumberArray('vtargetobjy',targetlisty)
    vtargetobj.putNumberArray('vtargetobjw',targetlistw)
    vtargetobj.putNumberArray('vtargetobjh',targetlisth)

# cube detection

    lower_yellow = np.array([30,100,100])
    upper_yellow = np.array([65,230,200])

    mask_yellow = cv2.inRange(gBlurImg, lower_yellow, upper_yellow)
    res_yellow = cv2.bitwise_and(frame,frame,mask=mask_yellow)
    res_yellow = cv2.dilate(res_yellow, np.ones((9,9),np.uint8), iterations=1)
    res_yellow = cv2.erode(res_yellow, np.ones((9,9),np.uint8), iterations=1)
    imgray_yellow = cv2.cvtColor(res_yellow, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',imgray_yellow)
    ret_yellow, thresh_yellow = cv2.threshold(imgray_yellow,127,255,0)

    contours_yellow, hierarchy_yellow = cv2.findContours(thresh_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cubelistx=[]
    cubelisty=[]
    cubelistw=[]
    cubelisth=[]
    cubelistpixels=[]
    if len(contours_yellow) !=0:
	i=0
	while (i<len(contours_yellow)):
		if(cv2.contourArea(contours_yellow[i])>minpixels_cube):
			comp=hierarchy_yellow[0,i,3]
			if (comp==-1):
				x,y,w,h =cv2.boundingRect(contours_yellow[i])
				roi=imgray_yellow[x:x+w,y:y+h]
				pixels=w*h-cv2.countNonZero(roi)
				cubelistpixels.append(pixels)
				cubelistx.append(x)
				cubelisty.append(y)
				cubelistw.append(w)
				cubelisth.append(h)
				cv2.rectangle(res_yellow, (x,y), (x+w, y+h), (0,0,255),2)
		i=i+1

    cv2.imshow('cubes',res_yellow)
    cubetarget.putNumber('objcount', len(cubelistx))
    cubetarget.putNumberArray('cubelistpixels',cubelistpixels)
    cubetarget.putNumberArray('cubetargetx',cubelistx)
    cubetarget.putNumberArray('cubetargety',cubelisty)
    cubetarget.putNumberArray('cubetargetw',cubelistw)
    cubetarget.putNumberArray('cubetargeth',cubelisth)




    if cv2.waitKey(1) & 0xFF == ord('p'):
	print p
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

statustable2 = NetworkTables.getTable("status")
statustable2.putBoolean('booted', False)
sys_bus = dbus.SystemBus()
hal_srvc = sys_bus.get_object('org.freedesktop.Hal',
                              '/org/freedesktop/Hal/devices/computer')
pwr_mgmt =  dbus.Interface(hal_srvc,
                'org.freedesktop.Hal.Device.SystemPowerManagement')
shutdown_method = pwr_mgmt.get_dbus_method("Shutdown")
shutdown_method()
vid.release()
cv2.destroyAllWindows()
