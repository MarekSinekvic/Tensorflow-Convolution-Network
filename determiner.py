import os
import datetime
import numpy as np
import shutil
import sys
from io import BytesIO
import random

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.python.client import device_lib

import tensorflow.keras.regularizers as regularizers

import PIL.Image as Image

TARGET_PATH = r"C:\UserFiles\Coding\NeuralNetworks\TensorFlowTest"
INPUTS_PATH = TARGET_PATH+r"\Inputs"
TENSORBOARD_LOGS_PATH = r"C:\UserFiles\Coding\NeuralNetworks\TensorFlowTest\logs"

def clearFolder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

print("Start\n\n")

clearFolder(TENSORBOARD_LOGS_PATH)
writer = tf.summary.create_file_writer(TENSORBOARD_LOGS_PATH)
# print("\n\n\n")
# with writer.as_default():
#     ind = 0
#     for slice, labels in dataset:
#         #print(tf.multiply(tf.cast(slice,tf.float16),1/255))

#         tf.summary.image("img2",[tf.multiply(tf.cast(slice,tf.float16)[1],1/255)],step=0)
#         ind+=1

# normLayer = tf.keras.layers.Rescaling(1./255)
# normalized_ds = dataset.map(lambda x, y : (normLayer(x),y))
# for slice, labels in normalized_ds:
#     print(slice,labels)

dropout = 0.1
Model = None
GeneratorModel = None

def getImageFrom(path, size = (400,400)):
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize(size)
    return np.array(img,dtype='float16')/255
def parseFolderAsDataset(path, size = (224,224), count = np.inf):
    if path[len(path)-1] != '\\' and path[len(path)-1] != '/':
        path += '/'

    dataset = [[],[]]
    classes = os.listdir(path)
    ind = 0
    for _class in classes:
        images = os.listdir(path+_class)
        imgInd = 0
        for image in images:
            if imgInd > count: continue
            full_path = path+_class+"/"+image

            img = Image.open(full_path)
            img = img.convert("RGB")
            img = img.resize(size)

            dataset[0].append(np.array(img,dtype='float16')/255)
            dataset[1].append(ind)
            
            imgInd += 1
        ind += 1
    
    #print(np.array(dataset[0]))
    #print(np.array(dataset[1]).shape)
    return dataset

def trainModel(epochsCount = 30, learningRate = 1e-4, l1=0, l2=0,continueTrain=False):
    global Model

    print((epochsCount, learningRate, l1, l2))

    #dataset = tf.keras.preprocessing.image_dataset_from_directory(INPUTS_PATH, image_size=(224,224), batch_size=1024)
    dataset = parseFolderAsDataset(INPUTS_PATH, (400,400))
    # print(list(dataset.as_numpy_iterator()))
    # print(list(dataset.as_numpy_iterator())[0])
    # print(list(dataset.as_numpy_iterator())[0][0])
    # print(list(dataset.as_numpy_iterator())[0][0][0])

    if (not continueTrain):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(400,400,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.DepthwiseConv2D(3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(16, 1, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.DepthwiseConv2D(3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(32, 1, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.DepthwiseConv2D(3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(64, 1, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.DepthwiseConv2D(3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(64, 1, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.DepthwiseConv2D(3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(128, 1, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.DepthwiseConv2D(3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(92, 1, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.DepthwiseConv2D(3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(64, 1, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        # model = tf.keras.Sequential([
        #     tf.keras.Input(shape=(400-10,400-10,3)),
        #     tf.keras.layers.Conv2D(16, 3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Conv2D(8, 3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Conv2D(8, 3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Conv2D(2, 1, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     tf.keras.layers.Dense(2, activation='softmax')
        # ])
        print("New model")
    else:
        model = Model
        print("Continue model train")
    #print("dropout: "+ str(dropout))
    # model = tf.keras.applications.MobileNetV3Small()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.fit(np.array(dataset[0]),np.array(dataset[1]),epochs=epochsCount)
    model.summary()
    # model.fit(np.array([list(dataset.as_numpy_iterator())[0][0][0]]), np.array([0]),epochs=epochsCount)

    #print(model.predict(np.array([list(dataset.as_numpy_iterator())[0][0][0]])))
    # print(model.predict(np.array([list(dataset.as_numpy_iterator())[0][1][0]])))
    print("Train over")

    os.replace("SavedModel.keras","_SavedModel.keras")
    model.save("SavedModel.keras", overwrite=True)
    
    Model = model
    return model

GeneratorSize = 10
def trainGeneratorModel(epochsCount = 30, learningRate = 1e-4, l1=0, l2=0, continueTrain=False):
    global GeneratorModel
    
    dataset = parseFolderAsDataset(INPUTS_PATH, (400,400), 3)
    
    if not continueTrain:
        # model = tf.keras.Sequential([
        #     tf.keras.Input(shape=(400,400,3)),
        #     tf.keras.layers.Conv2D(16,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     tf.keras.layers.Conv2D(32,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     tf.keras.layers.Conv2D(64,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     tf.keras.layers.Conv2D(128,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     tf.keras.layers.Conv2D(3,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2)),
        #     Model
        # ])
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(400,400,3)),
            tf.keras.layers.Conv2D(16,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(16,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(64,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(64,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(128,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(32,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
            tf.keras.layers.Conv2D(3,3, activation='relu', kernel_regularizer=regularizers.l1_l2(l1,l2), padding="same"),
        ])
    else:
        model = GeneratorModel
        print("Ready model")
    model.summary()

    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
    #     loss= tf.losses.SparseCategoricalCrossentropy(), #MeanSquaredError
    #     metrics=['accuracy']
    # )

    # def trainGAN(image):
    #     randTensor = (tf.random.uniform(shape=(400,400,3)).numpy()*1)
    #     with tf.GradientTape() as discTape, tf.GradientTape() as genTape:
    #         genOut = model(randTensor, training=True)

    #         rDiscOut = Model(image,training=True)
    #         fDiscOut = Model(genOut,training=True)
            
    #         pass
            
    #     genGrad = genTape.gradient()

    inps = []
    for (ind,val) in enumerate(dataset[1]):
        randTensor = (tf.random.uniform(shape=(400,400,3)).numpy()*1).tolist() #2+GeneratorSize*GeneratorSize
        
        #print(randTensor)
        
        # randTensor[0] = 0
        # randTensor[1] = 0
        # randTensor[i] = 1

        inps.append(randTensor)
        # inps.append(dataset[0][ind])
    inps = np.array(inps)

    # model.fit(inps,np.array(dataset[1]),epochs=epochsCount)

    # outs = dataset[1]
    # ind = 0
    # for i in outs:
    #     # i = 1-i
    #     outs[ind] = 1
    #     ind+=1
    # outs = np.array(outs)
    outs = []
    for (i,val) in enumerate(dataset[1]):
        outs.append((np.array(tf.ones(shape=(2,)))*0).tolist())
        outs[i][0] = 0
        outs[i][1] = 1
        # outs[i][val] = 1

    # outs = np.array(outs)
    # print(outs)
    print(inps.shape)
    print("Inputs memory usage: "+str(sys.getsizeof(inps)/1024/1024) + " MB")
    
    print("train")
    # model.fit(inps,(outs),epochs=epochsCount,batch_size=2)
    def train_step(sample,labels):
        with tf.GradientTape() as tape, tf.GradientTape() as detTape:
            target = np.array([sample])
            genImage = model(target)
            # Image.fromarray(np.multiply(np.array(genImage[0]),255).astype(np.uint8)).save("GenerateImages/test"+str(len(os.listdir("GenerateImages")))+".png")
            logits = Model(genImage)
            loss = tf.keras.losses.CategoricalCrossentropy()(np.array(labels[ind]), logits)
            detLoss = tf.keras.losses.CategoricalCrossentropy()(1-np.array(labels[ind]), logits)

        grads = tape.gradient(loss,model.trainable_weights)
        detgrads = detTape.gradient(detLoss,Model.trainable_weights)
        tf.keras.optimizers.Adam(learning_rate=learningRate).apply_gradients(zip(grads, model.trainable_weights))
        tf.keras.optimizers.Adam(learning_rate=learningRate).apply_gradients(zip(detgrads, Model.trainable_weights))
        
        return (loss,detLoss)
    Model.trainable = True
    model.trainable = True
    for epoch in range(epochsCount):
        epochLoss = 0
        detEpochLoss = 0
        print("Epoch: " + str(epoch+1))
        for (ind,sample) in enumerate(inps):
            loss = train_step(sample,outs)
            epochLoss += np.array(loss[0]).tolist()
            detEpochLoss += np.array(loss[1]).tolist()
        print("Loss: " + (str(epochLoss)+", "+str(detEpochLoss)))
        genImage = model(np.array([inps[2]]))
        print(str(Model(genImage).numpy().tolist()[0]) + "\n")
        Image.fromarray(np.multiply(np.array(genImage[0]),255).astype(np.uint8)).save("GenerateImages/test"+str(len(os.listdir("GenerateImages")))+".png")

    print("Train ended")

    os.replace("SavedGeneratorModel.keras","_SavedGeneratorModel.keras")
    model.save("SavedGeneratorModel.keras", overwrite=True)
    GeneratorModel = model
    return model
def generateImage(target):
    inp = (tf.random.uniform(shape=(400,400,3)).numpy())
    # inp = getImageFrom("Inputs/class_1/img"+str(random.randint(0,100))+".jpg")

    # inp[0] = 0
    # inp[1] = 0
    # inp[target] = 1

    
    # randValue = str(random.randint(0,100))
    # target = "Inputs/class_"+str(target)+"/img"+randValue+".jpg"
    # inp = getImageFrom(target)
    # Image.open(target).save("GenerateImages/show"+randValue+".jpg")

    # pred = tf.keras.Model(inputs=GeneratorModel.inputs,outputs=[GeneratorModel.layers[len(GeneratorModel.layers)-1-1].output]).predict([inp.tolist()])[0]
    pred = GeneratorModel.predict([inp.tolist()])
    print(Model(pred).numpy().tolist()[0])
    pred = np.multiply(np.array(pred[0]),255).astype(np.uint8)

    return pred

def addImageToSummary(array, name="img",step=0,desc=""):
    #print([tf.multiply(tf.cast(array,tf.float16),1/255)])
    with writer.as_default():
        tf.summary.image(name,tf.multiply(tf.cast(array,tf.float16),1/255),max_outputs=1024,step=step,description=desc)
def addScalarToSummary(value, name="num",step=0):
    with writer.as_default():
        tf.summary.scalar(name,value,step=step)

def searchStringWith(arr, target):
    for s in arr:
        if target in s:
            return s
    return None


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())
print(tf.config.list_physical_devices('GPU'))

if os.path.exists("SavedModel.keras"):
    Model = tf.keras.models.load_model("SavedModel.keras") #
    print("Loaded model")
else:
    print("Model training")
    with writer.as_default():
        writer.flush()
    Model = trainModel()
    
if os.path.exists("SavedGeneratorModel.keras"):
    GeneratorModel = tf.keras.models.load_model("SavedGeneratorModel.keras")
    print("Loaded Generator")
else:
    print("Generator training")
    GeneratorModel = trainGeneratorModel()

with writer.as_default():
    writer.flush()

validationIteration = 0
#model.summary()

def validate(image):
    prediction = Model.predict(image)
    #predict_class = np.argmax([prediction[0][0],prediction[0][1]])
    predict_class = np.argmax(prediction[0])
    # print(((prediction[0][0],prediction[0][1],prediction[0][2],prediction[0][3],prediction[0][4])))
    print(prediction[0])
    print("Predicted class: " + str(predict_class))
    #validationIteration+=1
    return (prediction,predict_class)



validationImagesClass1 = []
validationImagesClass2 = []
def request(task):
    taskTokens = task.split(" ")
    if (taskTokens[0] == "validate"):
        if (not os.path.isfile(taskTokens[1])):
            print("File not found")
            return
        image = tf.keras.utils.load_img(taskTokens[1], target_size=(224,224))
        input_arr = tf.keras.utils.img_to_array(image)
        validate(np.array([input_arr]))

        validationImagesClass1.append(input_arr)
        print(len(validationImagesClass1))
        addImageToSummary(validationImagesClass1,"Validation images")

    if (taskTokens[0] == "validate-folder"):
        if (not os.path.isdir(taskTokens[1])):
            print("not folder")
            return
        for path in os.scandir(taskTokens[1]):
            if (not os.path.isfile(path.path)):
                print("File not found")
                return

            image = tf.keras.utils.load_img(path.path, target_size=(224,224))
            input_arr = tf.keras.utils.img_to_array(image)
            print("\n"+path.path)
            prediction = validate(np.array([input_arr]))

            if (prediction[1] == 0):
                validationImagesClass1.append(input_arr)
            if (prediction[1] == 1):
                validationImagesClass2.append(input_arr)

        addImageToSummary(validationImagesClass1,"Validation class 1")
        addImageToSummary(validationImagesClass2,"Validation class 2")
    if (taskTokens[0] == "rename-folder"):
        pass

    if (taskTokens[0] == "train"):
        with writer.as_default():
            writer.flush()
        if len(taskTokens) > 1:
            epochsCount = int(taskTokens[1])
        
        dropoutSetter = searchStringWith(taskTokens,"dropout")
        if dropoutSetter is not None:
            dropout = float(dropoutSetter.split('=')[1])
        trainModel()
    if (taskTokens[0] == "close" or taskTokens[0] == "exit"):
        return
    if (taskTokens[0] == "clear"):
        with writer.as_default():
            writer.flush()

        validationImagesClass1.clear()
        validationImagesClass2.clear()
        # addImageToSummary([ [[[0,0,0,0]]] ],"Validation class 1")
        # addImageToSummary([ [[[0,0,0,0]]] ],"Validation class 2")
        
        clearFolder(TENSORBOARD_LOGS_PATH)
        writer = tf.summary.create_file_writer(TENSORBOARD_LOGS_PATH)

'''

while True:
    print("\n\n")
    print("Write task (validate C://...)")
    task = input()
    taskTokens = task.split(" ")
    if (taskTokens[0] == "validate"):
        if (not os.path.isfile(taskTokens[1])):
            print("File not found")
            continue
        image = tf.keras.utils.load_img(taskTokens[1], target_size=(224,224))
        input_arr = tf.keras.utils.img_to_array(image)
        validate(np.array([input_arr]))

        validationImagesClass1.append(input_arr)
        print(len(validationImagesClass1))
        addImageToSummary(validationImagesClass1,"Validation images")

    if (taskTokens[0] == "validate-folder"):
        if (not os.path.isdir(taskTokens[1])):
            print("not folder")
            continue
        for path in os.scandir(taskTokens[1]):
            if (not os.path.isfile(path.path)):
                print("File not found")
                continue

            image = tf.keras.utils.load_img(path.path, target_size=(224,224))
            input_arr = tf.keras.utils.img_to_array(image)
            print("\n"+path.path)
            prediction = validate(np.array([input_arr]))

            if (prediction[1] == 0):
                validationImagesClass1.append(input_arr)
            if (prediction[1] == 1):
                validationImagesClass2.append(input_arr)

        addImageToSummary(validationImagesClass1,"Validation class 1")
        addImageToSummary(validationImagesClass2,"Validation class 2")
    if (taskTokens[0] == "rename-folder"):
        pass

    if (taskTokens[0] == "train"):
        with writer.as_default():
            writer.flush()
        if len(taskTokens) > 1:
            epochsCount = int(taskTokens[1])
        
        dropoutSetter = searchStringWith(taskTokens,"dropout")
        if dropoutSetter is not None:
            dropout = float(dropoutSetter.split('=')[1])
        trainModel()
    if (taskTokens[0] == "close" or taskTokens[0] == "exit"):
        break
    if (taskTokens[0] == "clear"):
        with writer.as_default():
            writer.flush()

        validationImagesClass1.clear()
        validationImagesClass2.clear()
        # addImageToSummary([ [[[0,0,0,0]]] ],"Validation class 1")
        # addImageToSummary([ [[[0,0,0,0]]] ],"Validation class 2")
        
        clearFolder(TENSORBOARD_LOGS_PATH)
        writer = tf.summary.create_file_writer(TENSORBOARD_LOGS_PATH)

print("\n\nEND")
'''