import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import os
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


################## Parameters ####################################
path = "myData"
labelFile = 'label.csv'
batch_size_val = 50
steps_per_epoch_val = 2000
epochs_val = 10
ImageDimensions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2


#################### Importing of the image #########################
count = 0
images = []
ClassNo = []
mylist = os.listdir(path)
print('Total Classes Detected', len(mylist))
NoofClasses = len(mylist)
print("Importing Classes...")
for x in range(0, len(mylist)):
    myPiclist = os.listdir(path+"/"+str(count))
    for y in myPiclist:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        ClassNo.append(count)
    print(count, end="")
    count += 1
print(" ")
images = np.array(images)
ClassNo = np.array(ClassNo)

############################ Split Data ###########################
X_train, X_test, y_train, y_test = train_test_split(images, ClassNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
# X_train = Array of images to train
# y_train = CORRESPONDING CLASS ID

########## To check if the number of images matches to number of label for each data set ###########
print("Data Shapes")
print("Train", end=""); print(X_train.shape, y_train.shape)
print("Validation", end=""); print(X_validation.shape, y_validation.shape)
print("Test", end=""); print(X_test.shape, y_test.shape)
assert(X_train.shape[0] == y_train.shape[0])
assert(X_validation.shape[0] == y_validation.shape[0])
assert(X_test.shape[0] == y_test.shape[0])
assert(X_train.shape[1:] == ImageDimensions)
assert(X_validation.shape[1:] == ImageDimensions)
assert(X_test.shape[1:] == ImageDimensions)

########################## Read CSV file ####################
data = pd.read_csv(labelFile)
print("data shape", data.shape, type(data))

################### Display some samples images of all the classes ###################
num_of_samples = []
cols = 5
num_classes = NoofClasses
fig, axs = plt.subplot(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        X_selected = X_train[y_train == j]
        axs[j][i].imshow(X_selected[random.randint(0, len(X_selected)- 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j)+ "-"+row["Name"])
            num_of_samples.append(len(X_selected))

########################## Display a bar chart showing the no of samples for each category #############
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class Number")
plt.ylabel("Number of Images")
plt.show()

#################### Preprocessing the images ################################
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
cv2.imshow("Grayscale Images", X_train[random.randint(0, len(X_train) - 1)])

###################### Add a depth of 1 ###########
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

################### Augmentation of images: to make it more generic ###############
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10,)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

############## To show augmented images samples #########
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(ImageDimensions[0], ImageDimensions[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(X_train, NoofClasses)
y_validation = to_categorical(X_validation, NoofClasses)
y_test = to_categorical(X_test, NoofClasses)

####################### Convolution Neural Network Model ##############
def myModel():
    No_of_Filters = 60
    No_of_Nodes = 500
    Size_of_Filter1 = (5, 5)
    Size_of_Filter2 = (3, 3)
    Size_of_pool = (2, 2)
    model = Sequential()
    model.add((Conv2D(No_of_Filters, Size_of_Filter1, input_shape=(ImageDimensions[0], ImageDimensions[1], 1),
                      activation='relu')))
    model.add((Conv2D(No_of_Filters, Size_of_Filter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=Size_of_pool))

    model.add((Conv2D(No_of_Filters//2, Size_of_Filter2, activation='relu')))
    model.add((Conv2D(No_of_Filters // 2, Size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=Size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(No_of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NoofClasses, activation='softmax'))
    #Compile Model
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

############## Train model #################
model = myModel()
print(model.summary())

history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_size_val), steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                              validation_data=(X_validation, y_validation), shuffle=1)

############## Plot ######################
##### Loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.xlabel('epoch')
plt.title('loss')

###### accuracy
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.xlabel('epoch')
plt.title('accuracy')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score', score[0])
print('Test accuracy', score[1])

###### store the model as a pickle object ##########
pickle_out = open('model_trained.p', "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)
























