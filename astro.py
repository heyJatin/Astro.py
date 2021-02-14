# Importing modules to resize, convert and merge Images

import os
import numpy as np
import cv2
from PIL import Image, ImageChops
from astropy.io import fits
import img_scale

# Importing tensorflow modules to train model and preprocess images

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as imageprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

print('--> Modules loaded successfully')



# dimensions of our images
img_width, img_height = 606, 608

# Path to directory where final images needs to be detected after training
img_to_detect = 'data/detect'

# Name and path for weights to be saved
top_model_weights_path = 'bottleneck_fc_model.h5'

# Path to training and validation data
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# Other Training parameters, change according to your data
nb_train_samples = 32
nb_validation_samples = 32
epochs = 50
batch_size = 16

# Do not change
rootfolder={train_data_dir, validation_data_dir, img_to_detect}
item = ''
num=0

# This loop goes through your directory and converts all fits images to PNG format
# Conversion method inspired by https://github.com/psds075/fitstoimg
for root in rootfolder:

    for item in os.listdir(root):

        for asteroid_or_not in os.listdir(root+'/'+item):

            i = 1
            if not asteroid_or_not.endswith('.png'):

                for filename in os.listdir(root+'/'+item+'/'+asteroid_or_not):
                    print('--> Reading '+root+'/'+item+'/'+asteroid_or_not+'/'+filename)

                    if filename.endswith('.fits'):
                        image_data = fits.getdata(root+'/'+item+'/'+asteroid_or_not+'/'+filename)

                        if len(image_data.shape) == 2:
                            sum_image = image_data
                        else:
                            sum_image = image_data[0] - image_data[0]
                            for single_image_data in image_data:
                                sum_image += single_image_data

                        sum_image = img_scale.sqrt(sum_image, scale_min=0, scale_max=np.amax(image_data))
                        sum_image = sum_image * 200
                        im = Image.fromarray(sum_image)
                        if im.mode != 'L':
                            im = im.convert('L')
                            im = im.point(lambda x: 0 if x<12 else 255, '1')

                        im.save(root+'/'+item+'/'+str(i)+".png")
                        print('--> '+root+'/'+item+'/'+str(i)+".png CONVERTED TO PNG")
                        im.close()
                        i += 1

                # This loop denoises PNG images
                for filename in range(1,5):

                    image = cv2.imread(root+'/'+item+'/'+str(filename)+'.png')

                    kernel = np.ones((5,5),np.uint8)
                    denoised = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

                    cv2.imwrite(root+'/'+item+'/'+str(filename)+'.png',denoised)


                image1 = Image.open(root+'/'+item+'/'+'1.png')
                image2 = Image.open(root+'/'+item+'/'+'2.png')
                image3 = Image.open(root+'/'+item+'/'+'3.png')
                image4 = Image.open(root+'/'+item+'/'+'4.png')

                image12 = ImageChops.subtract(image1, image2)
                image21 = ImageChops.subtract(image2, image3)
                image34 = ImageChops.subtract(image3, image4)
                image43 = ImageChops.subtract(image4, image1)

                image = ImageChops.add(image12, image21)
                image = ImageChops.add(image, image34)
                image = ImageChops.add(image, image43)

                # Subtracts images from each other of same set, this will eleminate static objects.
                # Only moving objects will be visible clearly
                image.save(root+'/'+item+'/'+'sub.png')
                print('--> '+root+'/'+item+'/'+'sub.png SUBTRACTED PNGs')

                image = cv2.imread(root+'/'+item+'/'+'sub.png')

                kernel = np.ones((2,2),np.uint8)
                denoised = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
                erosion = cv2.erode(image,kernel,iterations = 2)

                image = cv2.resize(erosion, (0,0), fx=0.25, fy=0.25)

                num+=1
                cv2.imwrite(root+'/'+item+'/'+'r'+str(num)+'.png',image)

                # Removing unwanted images
                os.remove(root+'/'+item+'/'+'sub.png')
                os.remove(root+'/'+item+'/'+'1.png')
                os.remove(root+'/'+item+'/'+'2.png')
                os.remove(root+'/'+item+'/'+'3.png')
                os.remove(root+'/'+item+'/'+'4.png')


print('--> All images saved')


# Training model below

print('--> Starting to training...')

datagen = ImageDataGenerator(rescale=1. / 255)

# Using VGG16, imagenet
model = applications.VGG16(include_top=False, weights='imagenet')

# Data generator for training set
generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
bottleneck_features_train = model.predict_generator(
    generator, nb_train_samples // batch_size)
np.save(open('bottleneck_features_train.npy', 'wb'),
        bottleneck_features_train)

# Data generator for validation set
generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
bottleneck_features_validation = model.predict_generator(
    generator, nb_validation_samples // batch_size)
np.save(open('bottleneck_features_validation.npy', 'wb'),
        bottleneck_features_validation)

print('--> Data Generated')

train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
train_labels = np.array(
    [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
validation_labels = np.array(
    [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

# Structure of model
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))

# Saving weights
model.save_weights(top_model_weights_path)


print('--> Done trainig')

model1 = applications.VGG16(include_top=False, weights='imagenet')

# This loop will start detecting Asteroids in your images
for sets in os.listdir('data/detect/set_of_fits'):
    if sets.endswith('.png'):

        img = imageprocess.load_img('data/detect/set_of_fits/'+sets, target_size=(img_width, img_height))
        x = imageprocess.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        features = model1.predict(x)

        print('--> Detecting Asteroids in your images')

        astro_proba = model.predict_proba(features)

        classes = model.predict_classes(features)

        # 'classes' is 0 if no asteroid was detected and 1 if asteroids were detected
        if classes == 0:
            print(str(classes)+':'+sets+': No Asteroid Detected')
        else:
            print(str(classes)+':'+sets+': Asteroid Detected')
