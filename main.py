from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator


FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE = 32
MAXPOOL_SIZE = 2
BATCH_SIZE = 16
STEPS_PER_EPOCH = 20000//BATCH_SIZE
EPOCHS = 10

model = Sequential()
model.add(Conv2D(NUM_FILTERS,
                 (FILTER_SIZE, FILTER_SIZE),
                 input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                 activation="relu"))
model.add(MaxPool2D(pool_size=(MAXPOOL_SIZE, MAXPOOL_SIZE)))

model.add(Conv2D(NUM_FILTERS,
                 (FILTER_SIZE, FILTER_SIZE),
                 input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
                 activation="relu"))
model.add(MaxPool2D(pool_size=(MAXPOOL_SIZE, MAXPOOL_SIZE)))

# flatten
model.add(Flatten())

# fully connected layer
model.add(Dense(128, activation="relu"))

# for regulization
model.add(Dropout(0.5))

# output
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])


training_data_generator = ImageDataGenerator(rescale=1./255)

training_set = training_data_generator. \
                   flow_from_directory('C:\\Users\\ani\\PycharmProjects\\cnn\PetImages\\data\\Train',
                                       target_size=(INPUT_SIZE,INPUT_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='binary',
                                       )

model.fit_generator(training_set, steps_per_epoch = STEPS_PER_EPOCH,
                    epochs=EPOCHS, verbose=1)
