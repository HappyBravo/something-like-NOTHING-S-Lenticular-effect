# "https://github.com/Maxomm/nothing_filter/blob/main/streamlit_app.py"
import io

import numpy as np
# import streamlit as st
from PIL import Image, ImageOps


def add_gradient(section_array, slice_w, img_height):
    new_gradient = create_gradient(int(slice_w), img_height)
    combined_array = section_array * 0.9 + new_gradient * (1 - 0.9)
    return combined_array


def create_gradient(width, height):
    result = np.zeros((height, width, 3), dtype=np.float64)
    for i in range(3):
        result[:, :, i] = np.tile(np.linspace(255, 0, width), (height, 1))
    return result


def apply_filter(image_file, slice_amt = 8, 
                 flt_str = 0.6, gradient_enabled = 1):
    
    image = Image.open(image_file)

    fixed_image = ImageOps.exif_transpose(image)
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

        new_data = img_array[0:img_height, cur_idx:end_idx]
        new_data = add_gradient(new_data) if gradient_enabled else new_data
        new_image = np.hstack((new_image, new_data)) if i > 0 else new_data
        
    new_image = Image.fromarray(np.uint8(new_image))
    return new_image


# if __name__ == "__main__":
#     st.title("Nothing Filter")

#     up_file = st.file_uploader(
#         label="Image",
#         type=["jpg", "png"],
#         accept_multiple_files=False,
#         label_visibility="hidden",
#     )
#     demo_img = Image.open("demo_in.jpeg")
#     image = Image.open(up_file) if up_file is not None else demo_img

#     IMG_FORMAT = str(image.format).lower()
#     fixed_image = ImageOps.exif_transpose(image)
#     converted_image = fixed_image.convert("RGB")
#     img_array = np.asarray(converted_image)

#     left_column, right_column = st.columns(2)
#     with left_column:
#         with st.expander("Change Parameters"):
#             slice_amt = st.slider("Slice Amount", 1, 10, 8) + 1
#             flt_str = st.slider("Filter Strength", 0.0, 1.0, 0.6, 0.1)
#             gradient_enabled = st.checkbox("Enable Gradient", value=True)

#     img_height, img_width = fixed_image.height, fixed_image.width
#     slice_w = img_width / slice_amt
#     s_idx = (img_width / 2) - (slice_w / 2)

#     filtered_img = apply_filter(img_array)
#     image = Image.fromarray(np.uint8(filtered_img))

#     with right_column:
#         st.image(image)
#         buf = io.BytesIO()
#         image.save(buf, format=IMG_FORMAT)
#         st.download_button(
#             label="Download",
#             data=buf.getvalue(),
#             file_name="nothing_filter." + IMG_FORMAT,
#         )