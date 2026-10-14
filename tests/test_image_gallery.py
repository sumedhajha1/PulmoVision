from pathlib import Path

from PIL import Image

from src.dataset_loader import list_supported_image_files


def test_tracked_gallery_contains_decodable_normal_and_pneumonia_images():
    gallery_dir = Path(__file__).resolve().parents[1] / "assets" / "gallery"

    normal_images = list_supported_image_files(gallery_dir / "NORMAL", limit=1)
    pneumonia_images = list_supported_image_files(gallery_dir / "PNEUMONIA", limit=1)

    assert normal_images
    assert pneumonia_images

    for image_path in [normal_images[0], pneumonia_images[0]]:
        with Image.open(image_path) as image:
            image.verify()
