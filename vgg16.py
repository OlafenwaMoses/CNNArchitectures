from keras.models import Model, Sequential
from keras.layers import Conv2D, Input, Dense, MaxPool2D, Flatten, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ModelCheckpoint
import numpy as np
import operator


# IdenProf dataset https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip
train_dir = "idenprof/train"
test_dir = "idenprof/test"

# pre-trained VGG16 model on Idenprof dataset
# https://github.com/OlafenwaMoses/CNNArchitectures/releases/download/v1/vgg16_model_idenprof_010-0.638.h5
model_path = "vgg16_model_idenprof_010-0.638.h5"
num_classes = 10
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

def vgg_16(input_shape):

    input = Input(shape=input_shape)

    network = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(input)
    network = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)

    network = MaxPool2D(pool_size=(2,2), strides=(2,2))(network)

    network = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)
    network = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)

    network = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(network)

    network = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)
    network = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)
    network = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)

    network = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(network)

    network = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)
    network = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)
    network = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)

    network = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(network)

    network = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)
    network = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)
    network = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu")(network)

    network = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(network)

    network = Flatten()(network)

    network = Dense(units=4096, activation="relu")(network)
    network = Dense(units=4096, activation="relu")(network)
    network = Dense(units=num_classes, activation="softmax")(network)

    model = Model(input=input, output=network)

    return model


def train():

    batch_size = 16

    train_gen = ImageDataGenerator(
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator()

    train_generateor = train_gen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=batch_size, class_mode="categorical")
    test_generator = test_gen.flow_from_directory(test_dir, target_size=(224,224), batch_size=batch_size, class_mode="categorical")

    checkpoint = ModelCheckpoint("vgg16_model_{epoch:03d}-{val_acc}.h5",
                                 monitor="val_acc",
                                 save_best_only=True,
                                 save_weights_only=True)


    model = vgg_16(input_shape=(224,224,3))
    model.summary()
    model.compile(optimizer=SGD(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit_generator(train_generateor, epochs=20, validation_data=test_generator, steps_per_epoch=len(train_generateor), validation_steps=len(test_generator), callbacks=[checkpoint])


def predict_image(image_path):
    None

    model = vgg_16(input_shape=(224,224,3))
    model.load_weights(model_path)

    image = load_img(image_path, target_size=(224,224))
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
#predict_image("test-images/idenprof-3.jpg")