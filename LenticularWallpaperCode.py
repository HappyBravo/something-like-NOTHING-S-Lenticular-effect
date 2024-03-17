# "https://github.com/NoahCristino/Lenticular-Generator"

from PIL import Image
import sys
import os
import random
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
def combo(imgs, out):
    images = map(Image.open, imgs)
    widths, heights = zip(*(i.size for i in images))
    images = map(Image.open, imgs)
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(out)
    
def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)

def strips(image, pieces):
    gen = []
    im = Image.open(image)
    width, height = im.size
    mult = width/pieces # width/pieces - use x and height/width to calculate sizes based on pieces width/pieces = 120
    cache_folder = "./cache/"
    for i in range(pieces):
        x = 0+(i*mult)
        f_pathh = f"{cache_folder}{image.split('.')[0]+'_'+str(i)+'.png'}"
        crop(image, (x, 0, x+mult, height), f_pathh) #imagename_iteration.png
        gen.append(f_pathh)
    return gen

list1 = strips(image1, psz)
list2 = strips(image2, psz)
new = []
for i in range(len(list1)):
    new.append(list1[i])
    new.append(list2[i])
combo(new, "final.png")

for n in new:
    os.remove(n)