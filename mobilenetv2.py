import keras
from keras.layers import Conv2D, DepthwiseConv2D, Dense, add, Activation, BatchNormalization, AvgPool2D, Flatten
from keras.models import Model, Input
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import operator
import numpy as np


# IdenProf dataset https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip
train_dir = "idenprof/train"
test_dir = "idenprof/test"
image_dim = 224

# pre-trained VGG16 model on Idenprof dataset
# https://github.com/OlafenwaMoses/CNNArchitectures/releases/download/v1/mobilenet_v2_model_057-0.693.h5
model_path = "mobilenet_v2_model_057-0.693.h5"
class_dict = {
    "0": "chef",
    "1": "doctor",
    "2": "engineer",
    "3": "farmer",
    "4": "firefighter",
    "5": "judge",
    "6": "mechanic",
    "7": "pilot",
    "8": "police",
    "9": "waiter"
}

num_classes = len(class_dict.keys())
num_epochs = 100
learning_rate = 0.01


def relu6(x):
    return K.relu(x,max_value=6)

def lr_schedule(epoch):


    if epoch > int(num_epochs * 0.8):
        learning_rate = 0.0001
    elif epoch > int(num_epochs * 0.5):
        learning_rate = 0.001
    elif epoch > int(num_epochs * 0.3):
        learning_rate = 0.005
    else:
        learning_rate = 0.01

    return learning_rate


def bottle_neck(input, input_channels, output_channels, expansion, stride):

    output = Conv2D(filters=(input_channels * expansion), kernel_size=1, strides=1, padding="same")(input)
    output = BatchNormalization()(output)
    output = Activation(relu6)(output)

    output = DepthwiseConv2D(kernel_size=(3,3), strides=stride, padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation(relu6)(output)

    output = Conv2D(filters=output_channels, kernel_size=1, strides=1, padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation(relu6)(output)

    if( stride == 1 and input_channels == output_channels):
        output = add([input, output])
    return output

def MobileNetV2(input_shape, num_classes):

    input = Input(shape=input_shape)
    network = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding="same")(input)
    network = BatchNormalization()(network)
    network = Activation(relu6)(network)

    network = bottle_neck(network, 32, 16, 1, 1)

    network = bottle_neck(network, 16, 24, 6, 2)
    network = bottle_neck(network, 24, 24, 6, 1)

    network = bottle_neck(network, 24, 32, 6, 2)
    network = bottle_neck(network, 32, 32, 6, 1)
    network = bottle_neck(network, 32, 32, 6, 1)

    network = bottle_neck(network, 32, 64, 6, 2)
    network = bottle_neck(network, 64, 64, 6, 1)
    network = bottle_neck(network, 64, 64, 6, 1)
    network = bottle_neck(network, 64, 64, 6, 1)

    network = bottle_neck(network, 64, 96, 6, 1)
    network = bottle_neck(network, 96, 96, 6, 1)
    network = bottle_neck(network, 96, 96, 6, 1)

    network = bottle_neck(network, 96, 160, 6, 2)
    network = bottle_neck(network, 160, 160, 6, 1)
    network = bottle_neck(network, 160, 160, 6, 1)

    network = bottle_neck(network, 160, 320, 6, 1)

    network = Conv2D(kernel_size=(1,1), strides=(1,1), padding="same", filters=1280)(network)
    network = BatchNormalization()(network)
    network = Activation(relu6)(network)
    network = AvgPool2D(pool_size=(7,7))(network)
    network = Flatten()(network)
    network = Dense(units=10, activation="softmax")(network)

    model = Model(inputs=input, outputs=network)

    return model


def train():

    batch_size = 16
    train_gen = ImageDataGenerator(
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator()


    train_generator = train_gen.flow_from_directory(train_dir, target_size=(image_dim, image_dim), class_mode="categorical")
    test_generator = test_gen.flow_from_directory(test_dir, target_size=(image_dim, image_dim), class_mode="categorical")

    checkpoint = ModelCheckpoint(filepath="mobilenet_v2_model_{epoch:03}-{val_acc}",
                                 monitor="val_acc",
                                 save_weights_only=True,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    model = MobileNetV2(input_shape=(image_dim, image_dim, 3), num_classes=num_classes)
    model.compile(optimizer=SGD(lr=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit_generator(train_generator, epochs=num_epochs, validation_data=test_generator,
                        steps_per_epoch=len(train_generator), validation_steps=len(test_generator),
                        callbacks=[checkpoint, lr_scheduler])


def predict_image(image_path):
    None

    model = MobileNetV2(input_shape=(image_dim,image_dim,3), num_classes=num_classes)
    model.load_weights(model_path)

    image = load_img(image_path, target_size=(image_dim,image_dim))
    image = img_to_array(image, data_format="channels_last")
    image = np.expand_dims(image, axis=0)

    batch_results = model.predict(image, steps=1)

    predictions, probabilities = decode_predictions(batch_results, top=5)

    for pred, probability in zip(predictions, probabilities):
        print(pred, " : ", probability)

def decode_predictions(batch_results, top=1):

    result_dict = dict()
    prediction_array = []
    probability_array = []

    for results in batch_results:
        for i in range(len(results)):
            result_dict[class_dict[str(i)]] = results[i]

    result_dict = sorted(result_dict.items(), key=operator.itemgetter(1), reverse=True)

    for item in result_dict:
        prediction_array.append(item[0])
        probability_array.append(item[1])

    return prediction_array[:top], probability_array[:top]

#train()
predict_image("test-images/idenprof-1.jpg")





