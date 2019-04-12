import tensorflow as tf
import numpy as np
from datetime import datetime

model = tf.keras.applications.ResNet50(weights='imagenet')
model.summary()

test_images = []
for i in range(10):
    img_path = 'images/0' + str(i) + '.jpg'
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    test_images.append(x)


time_start = datetime.utcnow()
print(time_start)
for i in range(1000):
    preds = model.predict(test_images[i % 10])

time_end = datetime.utcnow()
print(time_end)
time_elapsed = time_end - time_start
print("Total time: ", time_elapsed.total_seconds())

## decode the results into a list of tuples (class, description, probability)
## (one such list for each sample in the batch)
#print('Predicted:', decode_predictions(preds, top=3)[0])
## Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
