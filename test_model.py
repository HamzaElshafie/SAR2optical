import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('generator150.h5')

# Define the root path and the SAR images path
rootpath = os.path.abspath('./')
path_sar = os.path.join(rootpath, 'content/SAR2optical/dataset/test/s1')
sar_list = os.listdir(path_sar)
sar_list.sort()

# Load and preprocess SAR images
imgs_sar = []
for img_path in sar_list:
    print(img_path)
    img_sar = img_to_array(load_img(os.path.join(path_sar, img_path), color_mode='grayscale', target_size=(256, 256)))
    imgs_sar.append(img_sar)
imgs_sar = np.array(imgs_sar) / 127.5 - 1.

# Generate fake images using the model
fake_A = model.predict(imgs_sar)
fake_A = fake_A * 0.5 + 0.5

# Save generated images
output_dir = os.path.join(rootpath, 'content/SAR2optical/dataset/test/generated_images')
os.makedirs(output_dir, exist_ok=True)
for i in range(len(fake_A)):
    plt.imsave(os.path.join(output_dir, f'out_{sar_list[i]}.jpg'), fake_A[i])

print("Generated images saved successfully.")
