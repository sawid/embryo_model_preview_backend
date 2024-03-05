from PIL import Image
import requests
from io import BytesIO
import os

def blend_images(background_path_or_url, overlay_path, alpha=0.5, save_path="blended_image.png"):
    # If the background is a URL, fetch the image content
    if background_path_or_url.startswith("http"):
        response = requests.get(background_path_or_url)
        response.raise_for_status()  # Check if the request was successful
        background_image = Image.open(BytesIO(response.content))
    else:
        # Otherwise, assume it's a local file path
        background_image = Image.open(background_path_or_url)

    # Opening the secondary image (overlay image)
    overlay_image = Image.open(overlay_path)

    # Resize the overlay image to match the size of the primary image
    overlay_image = overlay_image.resize(background_image.size)

    # Convert both images to RGBA mode (assuming both are not in this mode)
    background_image = background_image.convert("RGBA")
    overlay_image = overlay_image.convert("RGBA")

    # Blend the images with the specified alpha value
    blended_image = Image.blend(background_image, overlay_image, alpha)

    # Save the blended image
    blended_image.save(save_path)

    # Return the path to the saved image
    return os.path.abspath(save_path)

# Example usage with a URL:
background_image_url = "https://europeivf.com/assets/uploads/2022/06/europeivf.com-blastocysta-embryotransfer-a-kryokonzervace-blastocysta-1.png"
overlay_image_path = "heatmap_overlay.png"
result_image = blend_images(background_image_url, overlay_image_path, alpha=0.5)

# Displaying the result image
result_image.show()
