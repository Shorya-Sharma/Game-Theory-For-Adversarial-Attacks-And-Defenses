# Game Theory For Adversarial Attacks And Defenses
- CMU Introduction to Deep Learning Fall 2020
- Team: IDL.dll

## Attacks
- All code (`.ipynb` files) related to attack experiments is in the `attacks/` folder.
- Our FGSM and MI-FGSM implementations are in the [summarize/attack.py](https://github.com/effie-0/IDL-Project/blob/main/summarize/attack.py).
- Caveat:
  - We use Google Colab Pro to run the code. Some blocks are related to loading the gdrive folder.
  - There are some blocks related to the checkpoints of our pre-trained classifiers. If you are interested in getting these checkpoint data, please contact us.

## Random Initialization

## SAP
- All code (`.ipynb` files) related to SAP experiments is in the `jingxual/` folder.
- renet18.py is the file for building network without SAP moudle.
- renet18SAP.py is the file for building network with SAP moudle.
- SAP-like-networks.ipynb is the file for generating multiple SAP-like networks.

## Super-Resolution-Based Defense
- All code related to this part is in the `congzou/` folder.
- clean_images is the folder to store the original clean images.
- perturbed_images is the folder to store the adversarial attack images.
- denoised_images is the folder to store the wavelet denoised images.
- recoverd_images is the folder to store the super resolved images.
- attack.py is the file to generate attack samples.
- cifarresnet.py is the file for defense classification.
- components.py is the file for network components.
- edsr.py is the file for image super resolution using EDSR.
- denoising.py is the file for image wavelet denoising.
