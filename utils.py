import cv2
import numpy as np
# from tensorflow.keras.models import load_model
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# def initializePredictionModel():
#     model = load_model("resources/my_digit_model.h5")
#     return model


def preProcess(img):
    imgGray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    imgBlur= cv2.GaussianBlur(imgGray, (5, 5), 1) 
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold 

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0

    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx 
                max_area = area
    return biggest,max_area

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols=np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def thresholding(input_img,threshold=200,max_value=255, min_value=0):
    N,M=input_img.shape
    image_out=np.zeros((N,M),dtype=np.uint8)
        
    for i  in range(N):
        for j in range(M):
            if input_img[i,j]> threshold:
                image_out[i,j]=max_value
            else:
                image_out[i,j]=min_value
                
    return image_out

# def getPrediction(boxes,model):
#     result=[]
#     for image in boxes:
#         img = np.asarray(image)
#         img = img[4:img.shape[0]-4,4:img.shape[1]-4]
#         img=cv2.resize(img,(28,28))
#         img = thresholding(img)
#         img=img.reshape(1,28,28,1)
#         predictions = model.predict(img)
#         # ClassIndex = np.argmax(predictions, axis=1)
#         # probVal = np.amax(predictions)
#         # probVal = predictions[0]
#         ClassIndex = np.argmax(predictions,axis=-1)
#         probVal=np.amax(predictions)
#         print(ClassIndex,probVal)
#         if probVal>0.8:
#             result.append(ClassIndex[0])
#         else:
#             result.append(0)
#     return result

def getPrediction(boxes):
    result = []
    for image in boxes:
        # Convert the box to an image that Tesseract can process
        img = np.asarray(image)
        img = img[4:img.shape[0]-4, 4:img.shape[1]-4]  # Crop borders
        img = cv2.resize(img, (100, 100))  # Resize for better OCR performance
        img = thresholding(img)
        
        # Use pytesseract to recognize the digits
        text = pytesseract.image_to_string(img, config='--psm 6 digits')  # Only look for digits
        
        try:
            # Extract the first digit, if recognized correctly
            digit = int(text.strip()) if text.strip().isdigit() else 0
        except ValueError:
            digit = 0  # Handle cases where no valid digit is recognized
        
        result.append(digit)
    
    return result

def displayNumbers(img,numbers,color=(0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(9):
        for y in range(9):
            if numbers[(y*9)+x]!=0:
                cv2.putText(img,str(numbers[(y*9)+x]),
                            (x*secW+int(secW/2)-10,int((y+0.8)*secH)),cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2,color,2,cv2.LINE_AA)
    return img







def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])

    rowsAvailable = isinstance(imgArray[0],list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0),None,scale,scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(rows):
            imgArray[x] = cv2.resize(imgArray[x], (0,0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver