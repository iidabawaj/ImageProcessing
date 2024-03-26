# Image Processing Project
Implementation of Image Processing Techniques to enhance images from security cameras.

## Goal of the project
Employ image processing techniques in security
cameras to enhance security camera footage to prevent and
discourage criminal activities. Clearer and more detailed images
enable early detection of potential threats, allowing security
personnel to take quick actions and avoid risks effectively.

<img width="175" alt="image" src="https://github.com/iidabawaj/ImageProcessing/assets/139181626/3fea8f89-1bea-4590-95e2-7a65a2db5032">

<img width="174" alt="image" src="https://github.com/iidabawaj/ImageProcessing/assets/139181626/7581d143-ef5b-40c8-985c-57da284c89cb">

### Methods used:
* Function fastNlMeansDenoisingColored from OpenCV library for Non-Local Means Denoising
* Function multiply from OpenCV library for enhancing colors by increasing the contrast
### Code:
'''
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the image with salt-and-pepper noise
image_path = '/content/parking.jpg'
image = cv2.imread(image_path)

# Apply Non-Local Means Denoising
denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Enhance color by increasing contrast
contrast_factor = 1.2
enhanced_color = cv2.multiply(denoised_image, np.array([contrast_factor]))

# Display the original, denoised, and enhanced color images using plt.subplot
plt.figure(figsize=(15, 5))

plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(132), plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)), plt.title('Denoised Image')
plt.subplot(133), plt.imshow(cv2.cvtColor(enhanced_color.astype(np.uint8), cv2.COLOR_BGR2RGB)), plt.title('Enhanced Color')

plt.show()
'''
