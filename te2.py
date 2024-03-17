import cv2
import numpy as np

# Load input images
image1 = cv2.imread("test.png")
image2 = cv2.imread("test.png")

def resize_image(image, scale_factor):
  # Resize image with specified scale factor
  new_width = int(image.shape[1] * scale_factor)
  new_height = int(image.shape[0] * scale_factor)
  resized_image = cv2.resize(image, (new_width, image.shape[0]), interpolation=cv2.INTER_AREA)
  return resized_image

def apply_blur(image, blur_kernel_size):
  # Convert image to uint8 if necessary
  if not image.dtype == np.uint8:
    image = image.astype(np.uint8)
  # Apply Gaussian blur with specified kernel size
  return cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size),cv2.BORDER_REFLECT)

# Get image dimensions
image_height, image_width, _ = image1.shape

# Define number of slices (adjust for smoothness)
num_slices = 35

# Calculate slice width
slice_width = image_width // num_slices

# Create empty list to store final image slices
final_image_slices = []

# Define blur kernel size (adjust for blur intensity)
blur_kernel_size = 33

for i in range(num_slices):
  # Get slice from each image
  slice1 = image1[:, i * slice_width:(i + 1) * slice_width]
  slice2 = image2[:, i * slice_width:(i + 1) * slice_width]

  # Apply blur to each slice
  blurred_slice1 = apply_blur(slice1, blur_kernel_size)
  blurred_slice2 = apply_blur(slice2, blur_kernel_size)

  # Interleave slices (alternate between images)
#   interleaved_slice = cv2.hconcat([slice1, slice2])
  interleaved_slice = cv2.hconcat([blurred_slice1, blurred_slice2])


  # Append interleaved slice to final image list
  final_image_slices.append(interleaved_slice)

# Concatenate final image slices horizontally
final_image = cv2.hconcat(final_image_slices)

final_image = resize_image(final_image, 0.5)

# Display or save the final image
# cv2.imshow("Lenticular Effect", final_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("final2.png", final_image)
#   print(f"Image saved as: {file")