from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
 
from keras.models import load_model
 
model = load_model('model_saved.h5')

# Variables (You need to change these according to yourself)
test_dir = 'test'
img_size = (800, 240)

test_datagen = ImageDataGenerator(rescale= 1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = img_size,
    batch_size = 1,
    class_mode = 'categorical',
    shuffle = False
)

class_indices = test_generator.class_indices
class_labels = list(class_indices.keys())

predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)

for i, prediction in enumerate(predictions):
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]

    true_class_index = int(test_generator.classes[i])
    true_class = class_labels[true_class_index]

    print(f"Image {i + 1}: Predicted class: {predicted_class}, True class: {true_class}")
