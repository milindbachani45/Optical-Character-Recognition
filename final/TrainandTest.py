import operator
import cv2
import numpy as np
import os
import argparse
#############Module level variables ########################
MIN_CONTOUR_AREA=100
RESIZED_IMAGE_WIDTH=20
RESIZED_IMAGE_HEIGHT=30
############################################################


class ContourWithDetails():
	#########instance variable ##################
	npaContour=None		#vaariable to store contour
	boundingRectangle=None	#variale to store bounding rectangle co-ordinates
	intRectX=None		#x co-ordinate of rectangle
	intRectY=None		#y co-ordinate of rectangle
	intRectWidth=None	#width of rectangle
	intRectHeight=None	#height of rectangle
	fltArea=None		#area of rectangle

	def calculateRectangle(self):
		[intX,intY,intW,intH]=cv2.boundingRect(self.npaContour)
		self.intRectX=intX
		self.intRectY=intY
		self.intRectWidth=intW
		self.intRectHeight=intH
		
	def validateContour(self):
		if self.fltArea<MIN_CONTOUR_AREA: return False
		return True

################## End of class ##########################

def main():
	contourWithData=[]	#declare empty list to store ContourWithDetails class 
	validContours=[]	#declare empty list to store valid ContourWithDetails class 
	
	try:
		npaClassifications=np.loadtxt("classifications.txt",np.float32)	#get the training data from files
		
	except:
		print "\nClassifications.txt not found.\n"
		os.system("pause")
		return

	try:
		npaFlattenedImages=np.loadtxt("flattened_images.txt",np.float32)	#get the  training data from files.
		
	except:
		print "\nflattened_images.txt not found.\n"
		os.system("pause")
		return
	
	npaClassifications=npaClassifications.reshape((npaClassifications.size,1))	#reshape the classificaions into 1d array for further use.

	kNearest=cv2.ml.KNearest_create()		#create the object of KNearest node.

	kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

	
	ap = argparse.ArgumentParser()			#parser object to read the commandline object.
	ap.add_argument("-i","--image",required= True ,help="Enter the training imoage.")	#add_argument method to take the input from commandline and apply all the constraints to it.

	args = vars(ap.parse_args()) #storing all the input at command line in args.

	testimg=cv2.imread(args["image"])
	if testimg is None:
		print "Image not found. Please enter the image path again\n\n"
		os.system("pause")
		return
	#end if

	imgGray = cv2.cvtColor(testimg, cv2.COLOR_BGR2GRAY)       # get grayscale image
	imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

                                                        # filter image from grayscale to black and white
    	imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)  
    	cv2.imshow("Threshold image",imgThresh)	                                  # constant subtracted from the mean or weighted mean
    	imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    	imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of 		finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points
    	cv2.imshow("Conters wali image",imgContours)
    	for npaContour in npaContours:                             # for each contour
        	contourData = ContourWithDetails()                                             # instantiate a contour with data object
        	contourData.npaContour = npaContour                                         # assign contour to contour with data
        	contourData.boundingRect = cv2.boundingRect(contourData.npaContour)     # get the bounding rect
        	contourData.calculateRectangle()                    # get bounding rect info
        	contourData.fltArea = cv2.contourArea(contourData.npaContour)           # calculate the contour area
        	contourWithData.append(contourData)                                     # add contour with data object to list of all contours with data
    	# end for
    	for contourWithData in contourWithData:                 # for all contours
        	if contourWithData.validateContour():             # check if valid
            		validContours.append(contourWithData)       # if so, append to valid contour list
        	# end if
    	# end for
	    
	validContours.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    	strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

 	for contourWithData in validContours:            # for each contour
                                                # draw a green rect around the current char
        	cv2.rectangle(testimg,                                        # draw rectangle on original testing image
        	              (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
        	              (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
        	              (0, 255, 0),              # green
        	              2)                        # thickness

        	imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
        	                   contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
	
        	imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        	npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        	npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

        	retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

        	strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

        	strFinalString = strFinalString + strCurrentChar            # append current char to full string
    	# end for
	print "\n" + strFinalString + "\n"                  # show the full string

	cv2.namedWindow("Test image",cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Test image",600,600)
    	cv2.imshow("Test image", testimg)      # show input image with green boxes drawn around found digits
    	cv2.waitKey(0)                                          # wait for user key press

    	cv2.destroyAllWindows()             # remove windows from memory

    	return

###################################################################################################
if __name__ == "__main__":
    main()
# end if

   
