from  sklearn.model_selection import train_test_split
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator


train_root = "C:\\Users\\ani\\PycharmProjects\\cnn\\PetImages\\data\\train\\"
test_root = "C:\\Users\\ani\\PycharmProjects\\cnn\\PetImages\\data\\test\\"

def train_test(folder_name):
  root, dir,  cats = next(os.walk(os.path.join(folder_name, "Cat")))
  rt_dog, _,  dogs = next(os.walk(os.path.join(folder_name, "Dog")))
  tran_cat, test_cat , train_dog, test_dog = train_test_split( cats, dogs, train_size=0.8)

  os.makedirs(os.path.join(".", "PetImages", "pet","test", cats))
  os.makedirs(os.path.join(".", "PetImages", "pet", "train", dogs))

  for cat in tran_cat:
    shutil.move(root+"\\"+cat , train_root+ "cat")

  for cat in test_cat:
    shutil.move(root+"\\"+cat , test_cat+"cat")

  for dog in train_dog:
    shutil.move(rt_dog+"\\"+dog , train_root+"dog")

  for dog in test_dog:
    shutil.move(rt_dog+"\\"+dog , test_root+"dog")


image_generator = ImageDataGenerator(rotation_range = 30,
                                     width_shift_range = 0.2,
                                     height_shift_range = 0.2,
                                     zoom_range = 0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')