import numpy as np
import cv2
import os
import string
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
import pickle

path = 'EnglishFnt/Fnt'

images = []
classNo = []
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32,32,3)

batch_SizeVal=50
epocVal= 2
stepsPerEpoch = 50

# Mendapatkan daftar huruf besar (A-Z) dan huruf kecil (a-z)
letters = string.digits + string.ascii_uppercase + string.ascii_lowercase

noOfClasses = len(letters)
print("Importing Classes ....")
for x in range(noOfClasses):
    folder_path = os.path.join(path, letters[x])
    if os.path.exists(folder_path):
        myPicList = os.listdir(folder_path)
        for y in myPicList:
            curImg = cv2.imread(os.path.join(folder_path, y))
            curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
            images.append(curImg)
            classNo.append(x)
        print(x, end = " ")
    else:
        print(f"Folder '{letters[x]}' not found.")

print("\n Total classes:", noOfClasses)

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
# print(classNo.shape)

# split data

X_train,X_test,y_train,y_test = train_test_split(images, classNo, test_size = testRatio)
X_train,X_validation, y_train, y_validation = train_test_split(X_train, y_train,  test_size = valRatio)
print("\n X Train: ",X_train.shape)
print("\nX Test: ",X_test.shape)
print("\nX Validation", X_validation.shape)

numOfSmples=[]
for x in range(0,noOfClasses):
    # print(len (np.where(y_train==x)[0]))
    numOfSmples.append(len(np.where(y_train==x)[0]))
print("\n num Of Samples", numOfSmples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),numOfSmples)
plt.title("No of Imaegs for ech Class")
plt.xlabel("Class ID")
plt.ylabel("Numebr of Images")
plt.show()
print(X_train[30].shape)

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(X_train[0])
# img = cv2.resize(img,(300,300))
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1],X_validation.shape[2],1)


dataGen = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   shear_range=0.1,
                                   rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)


def myModel():
    noOfFilters = 60
    sizeOfFilter1= (5,5)
    sizeOfFilter2= (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1, input_shape=(imageDimensions[0],
                                                              imageDimensions[1],
                                                              1),activation = 'relu'
                                                                )))
    model.add((Conv2D(noOfFilters,sizeOfFilter1,activation = 'relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2,activation = 'relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()

print(model.summary())

history = model.fit(dataGen.flow(X_train,y_train,
                                 batch_size=batch_SizeVal),
                                 steps_per_epoch = stepsPerEpoch,
                                 epochs=epocVal,
                                 validation_data=(X_validation, y_validation),
                                 shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Los')
plt.xlabel(['epoch'])

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.legend(['training'],['validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.legend()

plt.figure(3)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#plt.legend(['training'],['validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.legend()

plt.show()

score=model.evaluate(X_test,y_test,verbose=0)
print('score =', score[0])
print('accuracy =', score[1])

pickle_out = open("model_trained.h5", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

