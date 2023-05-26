import numpy as np
import cv2
import pickle
from keras.models import load_model

width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

pickle_in = open("model_trained.h5", "rb")
model = pickle.load(pickle_in)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (64, 64))
    img = preProcessing(img)
    cv2.imshow("Preprocessed image", img)
    img = img.reshape(1, 64, 64, 1)

    classIndex = model.predict_step(img)
    predictions = model.predict(img)
    probVal = np.amax(predictions)

    if probVal > 0.65:
        cv2.putText(imgOriginal, str(classIndex), (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



