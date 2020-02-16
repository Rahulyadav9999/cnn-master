from keras.models import load_model
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

model = load_model("C:\\Users\\ani\\PycharmProjects\\cnn\\"+"myfirstcnn.h5")

testing_data_generator = ImageDataGenerator(rescale = 1./255)

test_set = testing_data_generator. \
               flow_from_directory('C:\\Users\\ani\\PycharmProjects\\cnn\\PetImages\\data\\test',
                                   target_size=(INPUT_SIZE,INPUT_SIZE),
                                   batch_size=BATCH_SIZE,
                                   class_mode='binary')

score = model.evaluate_generator(test_set, steps=len(test_set))
for idx, metric in enumerate(model.metrics_names):
    print("{}: {}".format(metric, score[idx]))



