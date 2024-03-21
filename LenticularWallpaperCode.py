# "https://github.com/NoahCristino/Lenticular-Generator"

from PIL import Image, ImageFilter, ImageEnhance
import sys
import os
import random
import numpy as np
import math 
import cv2

from nothing_filter import *

MAX_ANGLE = 45

if len(sys.argv) != 4:
	sys.exit("Add the images as arguments and # of pieces!")

image1 = sys.argv[1]
image2 = sys.argv[2]
psz = int(sys.argv[3])
im = Image.open(image1)
w, h = im.size
im2 = Image.open(image2)
w2, h2 = im2.size
if w != w2 or h != h2:
    sys.exit("Images must be the same sizes!")

if psz > w:
    sys.exit("There are more lenticules than pixels!")

def perspective_transform(image, angle):
    # width, height = image.size
    height, width = image.shape[:2]

     # Define the points for perspective transformation
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    
    # Angle for perspective transformation
    angle_rad = math.radians(angle)
    rotation_matrix = np.float32([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                                  [math.sin(angle_rad), math.cos(angle_rad), 0],
                                  [0, 0, 1]])
    
    # Apply perspective transformation to get new points
    new_pts = np.dot(rotation_matrix, np.append(pts1.T, np.ones((1, 4)), axis=0)).T[:, :2]
    
    # Calculate the transformation matrix
    M = cv2.getPerspectiveTransform(pts1, new_pts)
    
    # Apply the perspective transformation
    transformed_image = cv2.warpPerspective(image, M, (width, height))
    
    return transformed_image

def combo(imgs, out):
    images = map(Image.open, imgs)
    widths, heights = zip(*(i.size for i in images))
    images = map(Image.open, imgs)
    total_width = sum(widths)
    max_height = max(heights)
    # max_height = max(heights) * 2  # Resizing height to 2 times

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        # resized_height = im.size[1] * 2
        # im_resized = im.resize((im.size[0], resized_height))
        # new_im.paste(im_resized, (x_offset, 0))

        new_im.paste(im, (x_offset, 0))
        # x_offset += int(np.floor(im.size[0]*((2/3))))
        x_offset += im.size[0]


    new_im.save(out)

def sine_wave_distortion_vert(image, amplitude=20, frequency=20):
    distorted_image = Image.new("RGB", image.size)
    pixels = distorted_image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            offset_x = int(amplitude * np.sin(2 * np.pi * frequency * y / height))
            pixels[x, y] = image.getpixel(((x + offset_x) % width, y))

    return distorted_image

def sine_wave_distortion_hori(image, amplitude=20, frequency=20):
    distorted_image = Image.new("RGB", image.size)
    pixels = distorted_image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            offset_x = int(amplitude * np.sin(2 * np.pi * frequency * x / width))
            pixels[x, y] = image.getpixel(((x + offset_x) % width, y))

    return distorted_image

def create_right_symmetry(image):
    """
    Create a right-symmetry reflection of the given image.
    :param image: The input image (PIL Image object)
    :return: Image with right-symmetry reflection applied
    """
    # Flip the image horizontally
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Merge the original and flipped images
    width, height = image.size

    ratioo = 1 +1

    symmetric_image = Image.new('RGB', (int(ratioo*width), height))
    symmetric_image.paste(image, (0, 0))
    symmetric_image.paste(flipped_image, (width, 0))

    symmetric_image = symmetric_image.resize((int(symmetric_image.size[0]/ratioo), symmetric_image.size[1]))

    return symmetric_image

def add_vignette(image, strength=2):
    """
    Add a vignette effect to the given image.
    :param image: The input image (PIL Image object)
    :param strength: Strength of the vignette effect (0-1)
    :return: Image with vignette effect applied
    """
    # Convert image to RGBA if it's not already in RGBA mode
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Create a black overlay with the same size as the image
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # # Create a gradient mask for the vignette effect
    # width, height = image.size
    # x_center, y_center = width // 2, height // 2
    # max_distance = max(x_center, y_center)

    # for y in range(height):
    #     for x in range(width):
    #         distance = ((x - x_center) ** 2 + (y - y_center) ** 2) ** 0.5
    #         alpha = int(255 * strength * (1 - distance / max_distance))
    #         overlay.putpixel((x, y), (255, 255, 255, 255 - alpha))

    # Define ellipse parameters for the vignette effect
    width, height = image.size
    x_center, y_center = width // 2, height // 2
    ellipse_radius_x = width // 2
    ellipse_radius_y = height // 2
    
    # Create a gradient mask for the vignette effect
    for y in range(height):
        for x in range(width):
            distance_x = abs(x - x_center) / ellipse_radius_x
            distance_y = abs(y - y_center) / ellipse_radius_y
            if distance_x <= 1 and distance_y <= 1:
                alpha = int(255 * strength * (1 - max(distance_x, distance_y)))
                overlay.putpixel((x, y), (255, 255, 255, 255 - alpha))

    # Apply the overlay to the image
    return Image.alpha_composite(image, overlay)

def apply_matte_effect(image, output_path="", 
                       blur_radius=5, brightness_factor=0.8, contrast_factor = 0.8, saturation_factor=0.5, alpha=0.2,
                       sharpness = 8.3):
    """
    Apply a matte effect to the given image and save the result.
    :param image: PIL Image object representing the input image.
    :param output_path: Path where the resulting image with matte effect will be saved.
    :param blur_radius: Radius of Gaussian blur filter (default is 5).
    :param brightness_factor: Factor to adjust brightness (default is 0.8).
    :param saturation_factor: Factor to adjust color saturation (default is 0.5).
    :param alpha: Transparency level of the overlay (default is 0.2).
    """
    blurred_image = image

    # # Enhance Sharpness 
    curr_sharp = ImageEnhance.Sharpness(image) 
    
    # Sharpness enhanced by a factor of 8.3 
    blurred_image = curr_sharp.enhance(sharpness)

    # # Step 1: Apply blur filter
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred_image = image.filter(ImageFilter.BoxBlur(radius=blur_radius))


    # Step 2: Adjust brightness and contrast
    enhancer = ImageEnhance.Brightness(blurred_image)
    blurred_image = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(image)
    blurred_image = enhancer.enhance(contrast_factor)


    # Step 3: Desaturate the image
    enhancer = ImageEnhance.Color(blurred_image)
    matte_image = enhancer.enhance(saturation_factor)

    # Step 4: Optionally, overlay a semi-transparent white layer
    # width, height = image.size
    # white_overlay = Image.new('RGB', (width, height), (255, 255, 255, 0))
    # matte_image = Image.blend(matte_image, white_overlay, alpha=alpha)

    return matte_image


def crop(image_path, coords, saved_location, slice_w, angle = 0, flip = 1):

    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    width, height = image_obj.size

    # Enhance Sharpness 
    curr_sharp = ImageEnhance.Sharpness(image_obj) 
    sharpness = 25
    # Sharpness enhanced by a factor of 8.3 
    image_obj = curr_sharp.enhance(sharpness)

    
    # # # 
    # < --- CHANGE THIS OPTION TO GET DIFFERENT EFFECT --- >
    # # # 
    # image_obj = image_obj.filter(ImageFilter.GaussianBlur(radius=30))
    # image_obj = image_obj.filter(ImageFilter.BoxBlur(radius=33))
  
    cropped_image = image_obj.crop(coords)
    c_width, c_height = cropped_image.size
    # cropped_image = image_obj.crop(coords)

    # Apply symmetry effect
    cropped_image = create_right_symmetry(cropped_image)

    cropped_image = Image.fromarray(add_gradient(np.asarray(cropped_image), 
                                                 slice_w = slice_w, 
                                                 img_height=c_height).astype(np.uint8))
    
    # # # 
    # < --- CHANGE THIS OPTION TO GET DIFFERENT EFFECT --- >
    # # # 
    # Apply sine wave distortion
    distorted_image = sine_wave_distortion_hori(cropped_image, amplitude=0.005*c_height*flip, frequency=1)
    cropped_image = distorted_image 
    
    
    # # # 
    # < --- CHANGE THIS OPTION TO GET DIFFERENT EFFECT --- >
    # # #  
    # cropped_image = cropped_image.filter(ImageFilter.GaussianBlur(radius=25))
    cropped_image = cropped_image.filter(ImageFilter.BoxBlur(radius=5))

    #  Apply vignette effect
    # cropped_image = add_vignette(cropped_image)

    # cropped_image.save(saved_location)
    
    # # # 
    # < --- CHANGE THIS OPTION TO GET DIFFERENT EFFECT --- >
    # # # 
    cropped_image = apply_matte_effect(cropped_image,
                                       brightness_factor=1.1, 
                                       saturation_factor=1.2, 
                                       contrast_factor=1.1,
                                       alpha=0.05,
                                       blur_radius=33,
                                       sharpness=20)
    
    cropped_image = cropped_image.resize((cropped_image.size[0]//2, cropped_image.size[1]))

    cropped_image.save(saved_location)

def strips2(image, slice_amt = 8, flt_str = 0.6, gradient_enabled = 1):
    gen = []

    im = Image.open(image)
    cache_folder = "./cache/"

    fixed_image = ImageOps.exif_transpose(im)

    img_height, img_width = fixed_image.height, fixed_image.width
    converted_image = fixed_image.convert("RGB")

    img_array = np.asarray(converted_image)

    new_image = img_array

    slice_w = img_width / slice_amt
    s_idx = (img_width / 2) - (slice_w / 2)

    for i in range(slice_amt):
        cur_idx = s_idx + (i - (slice_amt - 1) / 2) * slice_w * (1 - flt_str)
        cur_idx = int(cur_idx)
        end_idx = int(cur_idx + slice_w)
        flip = 1
        if i >= slice_amt//2:
            flip = -1

        f_pathh = f"{cache_folder}{image.split('.')[0]+'_'+str(i)+'.png'}" 
        crop(image, (cur_idx, 0, end_idx, img_height), f_pathh, flip=flip, slice_w = slice_w)
        gen.append(f_pathh)
    # gen = apply_filter(image)
    return gen 

def strips(image, pieces):
    gen = []
    im = Image.open(image)
    width, height = im.size
    mult = width/pieces # width/pieces - use x and height/width to calculate sizes based on pieces width/pieces = 120
    cache_folder = "./cache/"
    for i in range(pieces):
        # x = 0+(i*mult)
        x = 0+(i*mult)
        flip = 1
        if i >= pieces//2:
            flip = -1
        f_pathh = f"{cache_folder}{image.split('.')[0]+'_'+str(i)+'.png'}" 
        crop(image, (x, 0, x+mult, height), f_pathh, flip = flip, slice_w= mult) # imagename_iteration.png
        gen.append(f_pathh)
    return gen

# list1 = strips(image1, psz)
# list2 = strips(image2, psz)

filter_str = 0.25
list1 = strips2(image1, slice_amt=psz, flt_str=filter_str)
list2 = strips2(image2, slice_amt=psz, flt_str=filter_str)


new = []
for i in range(len(list1)):
    new.append(list1[i])
    new.append(list2[i])
combo(new[::1], "final.png")

for n in range(0, len(new)-1, 2):
    # print(n, new[n])
    os.remove(new[n])