from PIL import Image
import os

# List of image paths
folder_path = r"data_crop\val\1-2"  # Replace with your folder path
image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
target_size = (128, 32)             
background_color = (0, 0, 0, 0)

for path in image_paths:
    img = Image.open(path).convert("RGB")      # drop alpha channel
    # make a black canvas of 128×32
    new_img = Image.new('RGB', target_size, background_color)
    # center the original crop
    offset = ((target_size[0] - img.width) // 2,
              (target_size[1] - img.height) // 2)
    new_img.paste(img, offset)
    new_img.save(path, 'PNG')