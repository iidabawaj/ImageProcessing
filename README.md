# Image Processing Project
Implementation of Image Processing Techniques to enhance images from security cameras.

## Goal of the project
Employ image processing techniques in security
cameras to enhance security camera footage to prevent and
discourage criminal activities. Clearer and more detailed images
enable early detection of potential threats, allowing security
personnel to take quick actions and avoid risks effectively.

## Image 1
<img width="175" alt="image" src="https://github.com/iidabawaj/ImageProcessing/assets/139181626/3fea8f89-1bea-4590-95e2-7a65a2db5032">

<img width="174" alt="image" src="https://github.com/iidabawaj/ImageProcessing/assets/139181626/7581d143-ef5b-40c8-985c-57da284c89cb">

### Methods used
*  Function fastNlMeansDenoisingColored from OpenCV library for Non-Local Means Denoising.
* Function multiply from OpenCV library for enhancing colors by increasing the contrast.
### Code
```
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the image with salt-and-pepper noise
image_path = '/content/parking.jpg'
image = cv2.imread(image_path)

denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

contrast_factor = 1.2
enhanced_color = cv2.multiply(denoised_image, np.array([contrast_factor]))

# Display the original denoised image and the enhanced image using plt.subplot
plt.figure(figsize=(15, 5))

plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(132), plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)), plt.title('Denoised Image')
plt.subplot(133), plt.imshow(cv2.cvtColor(enhanced_color.astype(np.uint8), cv2.COLOR_BGR2RGB)), plt.title('Enhanced Color')

plt.show()
```
## Image 2
![image](https://github.com/iidabawaj/ImageProcessing/assets/139181626/1c9bc2ea-a945-41f9-8697-2921a35f858a)

### Methods used
* Function medianBlur from OpenCV library to remove salt and pepper noise.
* Function gamma correction to adjust the brightness and contrast to make the background visible.

### Code
```
image1 = plt.imread('img1.jpg', 0)

# adjust kernel size for the median filter
kernel_size=7

# Apply the median filter 
denoised_image = cv2.medianBlur(image1, kernel_size)

# Define the gamma value for gamma correction
gamma = 0.6  #gamma <1=lighten image, gamma >1 = darken image

# Apply gamma correction
def Gamma_Correction(image, gamma):
    # Normalize the image to the range [0, 1]
    normalized_image = image / 255.0

    corrected_image = np.power(normalized_image, gamma)

    # Scale the values back to the range [0, 255]
    adjusted_image = (corrected_image * 255).astype(np.uint8)

    return adjusted_image

gammaCorrection = Gamma_Correction(denoised_image, gamma)

# Display images
fig, axes = plt.subplots(1, 3,  figsize=(20, 10))
axes[0].imshow(image1)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(denoised_image)
axes[1].set_title('Median Filter (Filter = {})'.format(kernel_size))
axes[1].axis('off')


axes[2].imshow(gammaCorrection)
axes[2].set_title('Apply gammaCorrection(gamma = {})'.format(gamma))
axes[2].axis('off')

plt.show()
```

## Image 3
![image](https://github.com/iidabawaj/ImageProcessing/assets/139181626/61743882-07e8-49db-8a45-c691f0d973ab)

### Methods used 
* Function BilateralFilter from OpenCV library.
* Function Gamma correction to enhance the colors of the image.

### Code 
```
image2 = cv2.imread('/content/img2.jpg')

# Apply bilateral filtering
# d parameter specifies the size of the pixel neighborhood used during filtering
# A higher sigmaColor value allows more color variation within the neighborhood, resulting in a stronger smoothing effect.
# A lower sigmaColor value restricts the color variation, preserving edges and details better but providing less smoothing.
# A higher sigmaSpace value includes pixels farther away from the central pixel in the neighborhood, resulting in a stronger smoothing effect and potentially blurring edges and details more.
# A lower sigmaSpace value limits the spatial extent, preserving edges and details better but providing less smoothing.

smoothed_image = cv2.bilateralFilter(image2, d=11, sigmaColor=85, sigmaSpace=90)

# Convert the BGR images to RGB
smoothed_image_rgb = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB)

# Define the gamma value for gamma correction
gamma = 1.3
# Apply gamma correction

def Gamma_Correction(image, gamma):
    # Normalize the image to the range [0, 1]
    normalized_image = image / 255.0

    corrected_image = np.power(normalized_image, gamma)

    # Scale the values back to the range [0, 255]
    adjusted_image = (corrected_image * 255).astype(np.uint8)

    return adjusted_image
adjustLight=Gamma_Correction(smoothed_image_rgb, gamma)

# Display images
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

axes[0].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
axes[0].set_title('Pixelated Image')
axes[0].axis('off')

axes[1].imshow(Image.fromarray(smoothed_image_rgb))
axes[1].set_title('Using bilateral filtering technique')
axes[1].axis('off')

axes[2].imshow(adjustLight)
axes[2].set_title('Apply gammaCorrection(gamma = {})'.format(gamma))
axes[2].axis('off')

plt.show()
```

## Image 4
![image](https://github.com/iidabawaj/ImageProcessing/assets/139181626/dc320ce0-cae5-4ca0-8a90-60bfb92c9061)

### Methods used
* Function filter2D from OpenCV library for sharpening.

### Code
```
image4 = cv2.imread('/content/img4.jpg', cv2.IMREAD_UNCHANGED)

# Create a sharpening kernel
sharpen_filter = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

# Apply the sharpening kernel to the input image
sharp_image = cv2.filter2D(image4, -1, sharpen_filter)
sharp_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB)

# Display images
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(image4, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(sharp_image)
axes[1].set_title('Sharpened Image')
axes[1].axis('off')

plt.show()
```

## Image 5
![image](https://github.com/iidabawaj/ImageProcessing/assets/139181626/5e1430cd-b508-48d7-9f7d-2307d18eb67c)

### Methods used
* Function filter2D from OpenCV library for sharpening.
* Crop the image to the Region Of Interest (ROI).

### Code
```
image5 = cv2.imread('/content/img5.jpg')

# Create the sharpening kernel
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Sharpen the image
sharpened_image = cv2.filter2D(image5, -1, kernel)

# crop image
y=100
x=150
h=230
w=270
crop_image = sharpened_image[x:w, y:h]

# Display images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(image5, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')

axes[1].imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Sharpened Image')

axes[2].imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
axes[2].set_title('cropped Image')

plt.show()
```

## Image 6
![image](https://github.com/iidabawaj/ImageProcessing/assets/139181626/d0ba02cd-c7e5-4c68-9392-2672f6c29930)

### Methods used
* Crop the image to the ROI.
* GrabCut algorithm from OpenCV library to segment the foreground of an image from the background.

### Code
```
image_path = '/content/image12.jpg'
image = cv2.imread(image_path)

# Define the ROI coordinates for the person
x, y, h, w = 50, 210, 300, 560

# Crop the image to the ROI
roi_image = image[x:h, y:w]

# Create a mask for the background
mask = np.zeros(roi_image.shape[:2], np.uint8)

# Define background and foreground models
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Define the rectangle for GrabCut algorithm
rectangle = (10, 10, roi_image.shape[1] - 10, roi_image.shape[0] - 10)

# Apply GrabCut algorithm to update the mask
cv2.grabCut(roi_image, mask, rectangle, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Create a mask where the background is
background_mask = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')

# Apply the background mask to the cropped image
background_removed = roi_image * background_mask[:, :, np.newaxis]

# Create a white background image
white_background = np.ones_like(roi_image) * 255

# Combine the background-removed image with the white background
final_image = white_background + background_removed

# Display images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
axes[1].set_title("Cropped Image")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
axes[2].set_title("Modified Image")
axes[2].axis("off")

plt.show()
```

## Image 7
![image](https://github.com/iidabawaj/ImageProcessing/assets/139181626/6ce3654c-3e92-4699-a2ba-c4a47515f58f)

### Methods used
*  Function Histogram equalization (equalizeHist) from OpenCV to each channel.
*  Bi-linear Interpolation to improve resolution.
* Enhance_visibility function to adjust the contrast.

### Code
```
image = cv2.imread('/content/img13.jpg')

# Convert BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split the RGB channels
r, g, b = cv2.split(image_rgb)

# Apply histogram equalization to each channel
r_eq = cv2.equalizeHist(r)
g_eq = cv2.equalizeHist(g)
b_eq = cv2.equalizeHist(b)

# Adjust the green channel
green_adjusted = cv2.addWeighted(g_eq, 0.2, b_eq, 0.9, 0)

# Merge the adjusted channels back into an RGB image
adjusted_image_rgb = cv2.merge((r_eq, green_adjusted, b_eq))

# Apply Bi-linear Interpolation to improve resolution
upscale_factor = 2  # Adjust the upscale factor as needed
upsampled_image = cv2.resize(adjusted_image_rgb, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_LINEAR)

# Adjust the contrast using enhance_visibility function
def enhance_visibility(image, alpha, beta):

    adjusted_image = cv2.convertScaleAbs(image, alpha, beta)

    return adjusted_image

alpha=1
beta=1.2
enhance_contrast=enhance_visibility(upsampled_image, alpha, beta)

# Display images
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(upsampled_image)
axes[1].set_title("Adjusted Image")
axes[1].axis("off")

axes[2].imshow(enhance_contrast)
axes[2].set_title("enhanced Image")
axes[2].axis("off")
plt.show()
```

## Conclusion
In conclusion, the enhancement image project successfully
utilized image processing techniques to enhance the quality of
images captured from security cameras. By applying denoising,
deblurring, and resolution enhancement algorithms, the project
effectively reduced noise, improved sharpness, and increased
the level of detail in the images. These enhancements have
significant implications for various sectors, including
residential, commercial, and public spaces.
