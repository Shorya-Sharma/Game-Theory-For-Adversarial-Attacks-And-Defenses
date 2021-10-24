import matplotlib.pyplot as plt
import os
from skimage.restoration import denoise_wavelet
from keras.preprocessing import image

path_perturbed = 'perturbed_images'
path_denoised = 'denoised_images'
names_perturbed = os.listdir(path_perturbed)
num_images = len(names_perturbed)
eps = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]

for i in range(num_images):
    name_img = names_perturbed[i].split('.')[0]
    img = image.load_img(path_perturbed + '/' + name_img + '.png', target_size(32, 32))
    img_rgb = image.img_to_array(img)
    img_bayes = denoise_wavelet(img_rgb / 255, sigma=eps[1], mode='soft', method='BayesShrink', multichannel=True,
                                convert2ycbcr=True)
    plt.imsave(path_denoised + '/' + name_img + '_d.jpg', img_bayes)
