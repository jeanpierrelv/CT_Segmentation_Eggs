import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import zipfile

from sklearn.model_selection import train_test_split


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# U-NET
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
dataset_path = './data/'
dataset_name = 'kvasir-seg.zip'
path_to_zip_file = dataset_path + dataset_name
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(dataset_path)

dataset_path = './data/Kvasir-SEG'
# imagename_list = os.listdir(dataset_path + '/'+ 'images')
# masksname_list = os.listdir(dataset_path + '/'+ 'masks')

filenames = os.listdir(dataset_path + '/images')
filenames.sort()

image_list = []
masks_list = []

for filename in filenames:
    image_list.append(dataset_path + '/images/' + filename)
    masks_list.append(dataset_path + '/masks/' + filename)
    
train_input_img_paths, test_input_img_paths, train_target_mask_paths, test_target_mask_paths = train_test_split(image_list, masks_list, test_size=0.1, random_state=42)
train_input_img_paths, val_input_img_paths, train_target_mask_paths, val_target_mask_paths = train_test_split(train_input_img_paths, train_target_mask_paths, test_size=0.12, random_state=42)


def load_data(images_path, masks_path):
  samples = {'images': [], 'masks': []}

  for i in range(len(images_path)):
    img = plt.imread(images_path[i])
    mask = plt.imread(masks_path[i])
    img = cv2.resize(img, (256,256))
    masks = cv2.resize(mask, (256,256))

    # 332x487 resolution
    samples['images'].append(img)
    samples['masks'].append(masks)

  samples = {
        'images': np.array(samples['images']),
        'masks': np.array(samples['masks']),
  }

  return samples


train_samples = load_data(train_input_img_paths, train_target_mask_paths)
val_samples = load_data(val_input_img_paths, val_target_mask_paths)
test_samples = load_data(test_input_img_paths, test_target_mask_paths)

plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.imshow(train_samples['images'][10])
plt.subplot(1,2,2)
plt.imshow(train_samples['masks'][10])
plt.show()

def create_conv_block(input_tensor, num_filters):
  x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(input_tensor)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)

  return x



def create_unet(input_shape, num_filters=16, dropout=0.10): # 16 - 32 filters
  # Encoder
  c1 = create_conv_block(input_shape, num_filters * 1)
  p1 = tf.keras.layers.MaxPool2D((2,2))(c1)
  p1 = tf.keras.layers.Dropout(dropout)(p1)

  c2 = create_conv_block(p1, num_filters * 2)
  p2 = tf.keras.layers.MaxPool2D((2,2))(c2)
  p2 = tf.keras.layers.Dropout(dropout)(p2)

  c3 = create_conv_block(p2, num_filters * 4)
  p3 = tf.keras.layers.MaxPool2D((2,2))(c3)
  p3 = tf.keras.layers.Dropout(dropout)(p3)

  c4 = create_conv_block(p3, num_filters * 8)
  p4 = tf.keras.layers.MaxPool2D((2,2))(c4)
  p4 = tf.keras.layers.Dropout(dropout)(p4)

  c5 = create_conv_block(p4, num_filters * 16)

  # Decoder
  u6 = tf.keras.layers.Convolution2DTranspose(num_filters * 8, (3,3), strides=(2,2), padding='same')(c5)
  u6 = tf.keras.layers.concatenate([u6, c4])
  u6 = tf.keras.layers.Dropout(dropout)(u6)
  c6 = create_conv_block(u6, num_filters * 8)

  u7 = tf.keras.layers.Convolution2DTranspose(num_filters * 4, (3,3), strides=(2,2), padding='same')(c6)
  u7 = tf.keras.layers.concatenate([u7, c3])
  u7 = tf.keras.layers.Dropout(dropout)(u7)
  c7 = create_conv_block(u7, num_filters * 4)

  u8 = tf.keras.layers.Convolution2DTranspose(num_filters * 2, (3,3), strides=(2,2), padding='same')(c7)
  u8 = tf.keras.layers.concatenate([u8, c2])
  u8 = tf.keras.layers.Dropout(dropout)(u8)
  c8 = create_conv_block(u8, num_filters * 2)

  u9 = tf.keras.layers.Convolution2DTranspose(num_filters * 1, (3,3), strides=(2,2), padding='same')(c8)
  u9 = tf.keras.layers.concatenate([u9, c1])
  u9 = tf.keras.layers.Dropout(dropout)(u9)
  c9 = create_conv_block(u9, num_filters * 1)

  output = tf.keras.layers.Conv2D(3, (1,1), activation='sigmoid')(c9)
  model = tf.keras.Model(inputs=[input_shape], outputs= [output])

  return model


inputs = tf.keras.layers.Input((256, 256, 3))
model = create_unet(inputs)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    name="Adam"
)
model.compile(optimizer="Adam", loss='mse', metrics=['accuracy'])

model_history = model.fit(train_samples['images'], train_samples['masks'], validation_data=(val_samples['images'], val_samples['masks']), epochs=200, verbose=1)

plt.figure(figsize=(5,5))
plt.subplot(2,1,1)
plt.plot(model_history.history['loss'], label='training_loss')
plt.plot(model_history.history['val_loss'], label='validation_loss')
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(model_history.history['accuracy'], label='training_accuracy')
plt.plot(model_history.history['val_accuracy'], label='validation_accuracy')
plt.legend()
plt.grid(True)


def predict_test_samples(val_map, model):
  img = val_map['images']
  mask = val_map['masks']

  test_images = np.array(img)

  predictions = model.predict(test_images)

  return predictions, test_images, mask


def plot_images(test_image, predicted_mask, ground_truth):
  plt.figure(figsize=(20,20))

  plt.subplot(1, 3, 1)
  plt.imshow(test_image)
  plt.title('Image')

  plt.subplot(1, 3, 2)
  plt.imshow(predicted_mask)
  plt.title('Predicted mask')

  plt.subplot(1, 3, 3)
  plt.imshow(ground_truth)
  plt.title('Ground truth mask')
  plt.show()
  
predicted_masks, test_images, ground_truth_masks = predict_test_samples(val_samples, model)
plot_images(test_images[30], predicted_masks[30], ground_truth_masks[30])


predicted_masks, test_images, ground_truth_masks = predict_test_samples(test_samples, model)
plot_images(test_images[30], predicted_masks[30], ground_truth_masks[30])
