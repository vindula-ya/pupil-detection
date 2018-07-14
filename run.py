import cv2
import dlib
import numpy
import numpy as np
import imutils


#INITIALIZE DLIB FACE DETECTOR AND CREATE FACE LANDMARKS PREDICTOR
from imutils import face_utils

frontal_face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
# def eye_aspect_ratio(eye):
# 	# compute the euclidean distances between the two sets of
# 	# vertical eye landmarks (x, y)-coordinates
# 	A = scipy.euclidean(eye[1], eye[5])
# 	B = scipy.euclidean(eye[2], eye[4])
#
# 	# compute the euclidean distance between the horizontal
# 	# eye landmark (x, y)-coordinates
# 	C = scipy.euclidean(eye[0], eye[3])
#
# 	# compute the eye aspect ratio
# 	ear = (A + B) / (2.0 * C)
#
# 	# return the eye aspect ratio
# 	return ear

#GET INTERSECTION BETWEEN TWO LINES WITH COORDINATES ([x1,x2],[x2,y2])([x3,y3][x4,y4])
def get_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = float(y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    #if denom == 0 there is no slope, but in our case there will always be
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    #ub = float((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return (int(x), int(y))


def scale_faceangle(points, scale=1, offset=(0, 0)):
    mid = numpy.mean(points, axis=0)
    pts = []
    for i in range(len(points)):
        pts.append(
            tuple(
                numpy.array(
                    (numpy.subtract(
                        numpy.add(numpy.subtract(points[i], mid) * scale, mid),
                        offset)),
                    dtype=int)))
    return pts


def shape_to_np(shape):
    # initialize the list of (x, y)-coordinates
    coords = numpy.zeros((shape.num_parts, 2), dtype=int)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


while(True):
    # Capture frame-by-frame
    edgeTR, frame = cap.read()
    imageFrame = frame
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #DETECT FACES IN THE GRAYSCALE FRAME
    faces = frontal_face_detector(gray, 0)

    for face in faces:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = shape_predictor(gray, face)
        #PREDICT FACE LANDMARKS AND CONVERT THEM TO NUMPY ARRAY COORDINATES
        shape = shape_to_np(shape_predictor(gray, face))
        # print(shape)

        for i in range(1, 7):
            #LEFTEST EYE POINT
            # ("right_eye", (36, 42)),
	# ("left_eye", (42, 48)),
            eyeL = tuple(shape[36])
            eyeL2 = tuple(shape[37])
            eyeL3 = tuple(shape[38])
            eyeL4 = tuple(shape[39])
            eyeL5 = tuple(shape[40])
            eyeL6 = tuple(shape[41])

            #RIGHTEST EYE POINT
            eyeR = tuple(shape[45])
            #MIDDLE EYE POINT
            eyeM = tuple(shape[37])
            # print(eyeL,eyeR,eyeM)
            # chinB = tuple(shape[8])
            # tmp = numpy.subtract(numpy.mean((shape[6], shape[9]), axis=0),chinB)

            # cv2.circle(frame, (eyeL), 7, (255, 0, 0), -1)
            # cv2.circle(frame, (eyeL2), 7, (0, 255, 0), -1)
            # cv2.circle(frame, (eyeL3), 7, (0, 0, 255), -1)
            # cv2.circle(frame, (eyeL4), 7, (0, 255, 0), -1)
            # cv2.circle(frame, (eyeL5), 7, (255, 0, 0), -1)
            # cv2.circle(frame, (eyeL6), 7, (0, 0, 255), -1)

            # cv2.circle(frame, (eyeR), 7, (0, 255, 0), -1)
            # leftEyeHull = [eyeL,eyeL2,eyeL3,eyeL4,eyeL5,eyeL6]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_red = np.array([50,200,200])
            upper_red = np.array([255,255,255])

            mask = cv2.inRange(hsv, lower_red, upper_red)
            res = cv2.bitwise_or(frame,frame, mask= mask)
            kernel = np.ones((15,15),np.float32)/225
            smoothed = cv2.filter2D(res,-1,kernel)

            median = cv2.medianBlur(res,3)
            nota = cv2.bitwise_not(mask)
            # img_gray = cv2.cvtColor(nota, cv2.COLOR_BGR2GRAY)

            # cv2.imshow("skel",nota)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            im_in = cv2.imread("1.JPG", cv2.IMREAD_GRAYSCALE);

# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.

            th, im_th = cv2.threshold(nota, 220, 255, cv2.THRESH_BINARY_INV);

# Copy the thresholded image.
            im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
            h, w = im_th.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
            cv2.floodFill(im_floodfill, mask, (0,0), 255);

# Invert floodfilled image
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
            im_out = im_th | im_floodfill_inv
            nota2 = cv2.bitwise_not(im_out)
            a = cv2.bitwise_not(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
            kernel = np.ones((3,3),np.uint8)
# sure background area
            sure_bg = cv2.dilate(thresh,kernel,iterations=3)
            imageA = cv2.erode(sure_bg,kernel,iterations=8)
            imagenot = cv2.bitwise_not(imageA)

            img2 = nota2
            isolatebyOR = cv2.bitwise_or(gray,nota2)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(isolatebyOR, 75, 255, cv2.THRESH_BINARY)[1]
            imageA = cv2.dilate(thresh,kernel,iterations=1)
            imageA = cv2.erode(thresh,kernel,iterations=1)
            imagenot = cv2.bitwise_not(imageA)

            print(nota2)
            cnts = cv2.findContours(imagenot.copy(), cv2.RETR_EXTERNAL,
	        cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            loaction = list()
# print('start')
# loop over the contours
            i = 0
            for c in cnts:
	            i = i +1
	# compute the center of the contour
	# print('loop')
	            M = cv2.moments(c)
	            cX = int(M["m10"] / M["m00"])
	            cY = int(M["m01"] / M["m00"])

	# draw the contour and center of the shape on the image
	            cv2.drawContours(imageFrame, [c], -1, (0, 255, 0), 1)
	            cv2.circle(imageFrame, (cX, cY), 1, (0, 0, 255), 1)
                # cv2.putText(imageFrame, "center", (cX - 20, cY - 20),
		        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	# global loaction
	            loaction.append([cX, cY])


            # cv2.imshow('Original',frame)
            # cv2.imshow('Averaging',imageFrame)
            # pupilFrame = cv2.equalizeHist(frame[int(y + (h * .25)):int((y + h)), x:(x + w)])
            # pupilO = cv2.bitwise_not(pupilFrame)
            # cv2.line(frame, (eyeL), (eyeL2), (0, 0, 255), 1)
            # cv2.line(frame, (eyeL2), (eyeL3), (0, 0, 255), 1)
            # cv2.line(frame, (eyeL3), (eyeL4), (0, 0, 255), 1)
            # cv2.line(frame, (eyeL4), (eyeL5), (0, 0, 255), 1)
            # cv2.line(frame, (eyeL5), (eyeL6), (0, 0, 255), 1)
            # cv2.line(frame, (eyeL6), (eyeL), (0, 0, 255), 1)

            # mim = numpy.subtract(eyeL4,eyeL)
            # div = numpy.divide(mim,2)
            # inaa = numpy.round(div)
            # az = int(inaa)
            # print(az)
            # a = numpy.add(eyeL,eyeL4)
            # div = numpy.divide(a,2)
            # inaa = numpy.round(div,2)
            # zz = a//2
            # print(mim,zz)
            # b = tuple(zz)
            # cv2.circle(frame, (b), 7, (0, 255, 0), -1)



            # cv2.circle(frame, (eyeM), 7, (0, 0, 255), -1)
            # leftEAR = eye_aspect_ratio(eyeL)
            # rightEAR = eye_aspect_ratio(eyeR)

            # NOSE TOP POINT
            noseT = tuple(numpy.mean((numpy.mean((shape[21], shape[22]), axis=0), eyeM), axis=0))

            #NOSE BOTTOM POINT
            noseB = tuple(shape[33])
            #UPPER LIP BOTTOM MID POINT
            lipU = tuple(shape[62])
            #LOWER LIP TOP MID POINT
            lipL = tuple(shape[66])

            #CHIN BOTTOM POINT
            chinB = tuple(shape[8])
            # tmp = numpy.subtract(numpy.mean((shape[6], shape[9]), axis=0),chinB)
            # #CHIN LEFT POINT; CALCULATING MORE PRECISE ONE
            # chinL = tuple(numpy.subtract(numpy.mean((shape[6], shape[7]), axis=0),tmp))
            # #CHIN RIGHT POINT; CALCULATING MORE PRECISE ONE
            # chinR = tuple(numpy.subtract(numpy.mean((shape[9], shape[10]), axis=0),tmp))

            # #THE DIFFERENCE (eyeM - chinB) EQUALS 2/3 OF THE FACE
            # tmp = numpy.subtract(eyeM, chinB)
            # #GET 1/3 OF THE FACE
            # tmp = tuple([int(x / 2) for x in tmp])
            #
            # #CALCULATING THE EDGES FOR THE BOX WE ARE GOING TO DRAW
            # #EDGE POINT TOP LEFT, LEFT EYEBROW + 1/3 OF THE FACE SO WE GET THE FOREHEAD LINE
            # edgeTL = tuple(numpy.add(shape[19], tmp))
            # #EDGE POINT TOP RIGHT, RIGHT EYEBROW + 1/3 OF THE FACE SO WE GET THE FOREHEAD LINE
            # edgeTR = tuple(numpy.add(shape[24], tmp))
            #
            # #MOVE THE TOP LEFT EDGE LEFT IN LINE WITH THE CHIN AND LEFT EYE - ESTIMATING FOREHEAD WIDTH
            # edgeTL = get_intersection(edgeTL[0], edgeTL[1], edgeTR[0], edgeTR[1], eyeL[0],
            #                        eyeL[1], chinB[0], chinB[1])
            #
            # #MOVE THE TOP RIGHT EDGE RIGHT IN LINE WITH THE CHIN AND RIGHT EYE - ESTIMATING FOREHEAD WIDTH
            # edgeTR = get_intersection(edgeTR[0], edgeTR[1], edgeTL[0], edgeTL[1], eyeR[0],
            #                        eyeR[1], chinB[0], chinB[1])
            #
            # tmp = numpy.subtract(eyeM, chinB)
            #
            # #EDGE POINT BOTTOM LEFT, CALCULATE HORIZONTAL POSITION
            # edgeBL = tuple(numpy.subtract(edgeTL, tmp))
            # #EDGE POINT BOTTOM RIGHT, CALCULATE HORIZONTAL POSITION
            # edgeBR = tuple(numpy.subtract(edgeTR, tmp))
            #
            # #EDGE POINT BOTTOM LEFT, CALCULATE VERTICAL POSITION - IN LINE WITH CHIN SLOPE
            # edgeBL = get_intersection(edgeTL[0], edgeTL[1], edgeBL[0], edgeBL[1], chinL[0], chinL[1],
            #                        chinR[0], chinR[1])
            # #EDGE POINT BOTTOM RIGHT, CALCULATE VERTICAL POSITION - IN LINE WITH CHIN SLOPE
            # edgeBR = get_intersection(edgeTR[0], edgeTR[1], edgeBR[0], edgeBR[1], chinR[0], chinR[1],
            #                        chinL[0], chinL[1])
            #
            #CALCULATE HEAD MOVEMENT OFFSET FROM THE CENTER, lipU - lipL IS THE DISTANCE FROM BOTH LIPS (IN CASE MOUTH IS OPEN)
            offset = (float(noseT[0] - 2 * noseB[0] + chinB[0] + lipU[0]-lipL[0]) * 4,
                      float(noseT[1] - 2 * noseB[1] + chinB[1] + lipU[1]-lipL[1]) * 4)
            #
            # #BACKGROUND RECTANGLE
            # recB = (edgeTL, edgeTR, edgeBR, edgeBL)
            #
            # #FOREBACKGROUND RECTANGLE
            # recF = (scale_faceangle((recB), 1.5, offset))

            #DRAW FACIAL LANDMARK COORDINATES
            # for (x, y) in shape:
            #     cv2.circle(frame, (x, y), 1, (255, 0, 255), 5)

            # #DRAW BACKGROUND RECTANGLE
            # cv2.polylines(frame, numpy.array([recB], numpy.int32), True,
            #               (0, 0, 255), 5)

            # # DRAW FACE BOX EDGE LINES
            # for i in range(4):
            #     cv2.line(frame, recB[i], recF[i], (255, 255, 0), 5)

            #DRAW NOSE DIRECTION LINE
            # cv2.line(
            #     frame, tuple(shape[30]),
            #     tuple(
            #         numpy.array(
            #             (numpy.subtract(shape[30], offset)), dtype=int)),
            #     (0, 255, 255), 5)

            # #DRAW FOREGROUNDBACKGROUND RECTANGLE
            # cv2.polylines(frame, numpy.array([recF], numpy.int32), True,
            #               (0, 255, 0), 5)

    cv2.imshow('Frame', frame)

    #PRESS ESCAPE TO EXIT
    if cv2.waitKey(1) == 27:
        break

#RELEASE THE CAP
cap.release()
cv2.destroyAllWindows()
