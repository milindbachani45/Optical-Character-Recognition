import argparse
import cv2
import os
import numpy as np
import sys

#Module level variables (Global variables)###################################################
MIN_CONTOUR_AREA=100
RESIZED_IMAGE_WIDTH=20
RESIZED_IMAGE_HEIGHT=30
#############################################################################################


def main():
	ap = argparse.ArgumentParser()			#parser object to read the commandline object.
	ap.add_argument("-i","--image",required= True ,help="Enter the training imoage.")	#add_argument method to take the input from commandline and apply all the constraints to it.

	args = vars(ap.parse_args()) #storing all the input at command line in args.

	trainingimg=cv2.imread(args["image"])
	if trainingimg is None:
		print "Image not found. Please enter the image again\n\n"
		os.system("pause")
		return
	#end if

	imgGray=cv2.cvtColor(trainingimg,cv2.COLOR_BGR2GRAY)	#get grayscale image
	imgBlurr=cv2.GaussianBlur(imgGray,(5,5),0)		#get blurred image

	imgThresh = cv2.adaptiveThreshold(imgBlurr,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean
	cv2.namedWindow("Threshold image",cv2.WINDOW_NORMAL)	
	cv2.resizeWindow("Threshold image",600,600)	
	cv2.imshow("Threshold image",imgThresh)				#show the thresholded image
	imgThreshCopy=imgThresh.copy()					#make a copy of image as the original image will be modified by contor function
	imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,        # input image, make sure to use a copy since the function will modify this image in the course of 												 #finding contours
                                                 cv2.RETR_EXTERNAL,                 # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)           # compress horizontal, vertical, and diagonal segments and leave only their end points

						 # declare empty numpy array, we will use this to write to file later
                                		 # zero rows, enough cols to hold all image data
	npaFlattenedImages=np.empty((0,RESIZED_IMAGE_WIDTH*RESIZED_IMAGE_HEIGHT))

	intClassifications = []         # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end	

	                                # possible chars we are interested in are digits 0 through 9 and alphabets, put these in list intValidChars
        intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                         ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                         ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                         ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'),
		         ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'),
                         ord('k'), ord('l'), ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'),
                         ord('u'), ord('v'), ord('w'), ord('x'), ord('y'), ord('z')]
	count=0
	for npacontour in npaContours:						 #for each contour in contours 
		if cv2.contourArea(npacontour)>MIN_CONTOUR_AREA:		#check if the current contour is big enough to consider
			[intX,intY,intWidth,intHeight]=cv2.boundingRect(npacontour)	#extract the coordinates of the rectangles which fits the contour.
			count = count + 1
			                                      # draw rectangle around each contour as we ask user for input
            		cv2.rectangle(trainingimg,           # draw rectangle on original training image
                          	     (intX, intY),                 # upper left corner
                                     (intX+intWidth,intY+intHeight),        # lower right corner
                                     (0, 255,0),                  # green
                                      2)                            # thickness

			imgROI=imgThresh[intY:intY+intHeight,intX:intX+intWidth]			#crop the image out of thr threshold image
			imgROIResized=cv2.resize(imgROI,(RESIZED_IMAGE_WIDTH,RESIZED_IMAGE_HEIGHT))	#resize the inage in a stadardized formate so it can be stored for learning purpose.
			

			cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference
            		
			cv2.namedWindow("training image",cv2.WINDOW_NORMAL)
			cv2.resizeWindow("training image",600,600)
			cv2.imshow("training image", trainingimg)      # show training numbers image, this will now have red rectangles drawn on it

			intChar=cv2.waitKey(0)		#get key press
			print "Key stroke:%d"%intChar
			if intChar == 229:
				print "Caps lock pressed"
				intChar=cv2.waitKey(0)
			#end if
			if intChar == 27:
				sys.exit()
			elif intChar in intValidChars:
				intClassifications.append(intChar)  # append classification char to integer list of chars (we will convert to float later before writing to file)
				npaFlattenedImage=imgROIResized.reshape((1,RESIZED_IMAGE_WIDTH*RESIZED_IMAGE_HEIGHT)) # flatten image to 1d numpy array so we can write to file later
				npaFlattenedImages=np.append(npaFlattenedImages,npaFlattenedImage,0) # add current flattened impage numpy array to list of flattened image numpy arrays
			#end if else 
		#end of if 
		else: print "Noise detected" #if the contour is  less than the predetermined size we consider it as a noise.
	#end of for loop
	fltClassifications = np.array(intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats
	npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later
		
	print "\n\ntraining complete !!\n"
	print "%d images trained"%count
		
	if os.path.isfile("classifications.txt"):
		f_handle = file("classifications.txt", 'a')	
		np.savetxt(f_handle, npaClassifications)
		f_handle.close()
	else: np.savetxt("classifications.txt", npaClassifications)           # write flattened images to file	
	#end if

	if os.path.isfile("flattened_images.txt"):
		f_handle = file("flattened_images.txt", 'a')	
		np.savetxt(f_handle, npaFlattenedImages)
		f_handle.close()
	else: np.savetxt("flattened_images.txt", npaFlattenedImages)           # write flattened images to file	
	#end if
	cv2.destroyAllWindows()             # remove windows from memory
	return
#end of main method.
if __name__ == "__main__":
    main()
# end if

