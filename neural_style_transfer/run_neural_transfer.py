from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, Lambda

from src.losses import TotalLoss
from src.model import VGG19
from config import IMAGE_DIR


def run_neural_transfer(
        content_image_path: str,
        style_image_path: str,
):
    # LOAD IMAGES
    images = {
        'content_image': Image.open(content_image_path),
        'style_image': Image.open(style_image_path),
        'input_image': Image.open(content_image_path),  # Use content image as starting point
    }

    # PREPROCESS
    preprocess = Compose([
        ToTensor(),
        Resize(size=(224, 224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Lambda(lambda x: x.unsqueeze(0)),  # Add batch size
    ])
    images = {name: preprocess(image) for name, image in images.items()}

    # MODEL
    content_layers = [
        21   # conv4_2
    ]
    style_layers = [
        0,   # conv1_1
        5,   # conv2_1
        10,  # conv3_1
        19,  # conv4_1
        28,  # conv5_1
    ]
    model = VGG19(
        content_layers=content_layers,
        style_layers=style_layers,
    )

    # LOSS FUNCTION
    _, content_features, _ = model(images['content_image'])
    _, _, style_features = model(images['style_image'])

    loss_criterion = TotalLoss(content_features.values(), style_features.values())

    # Style transfer


if __name__ == '__main__':
    run_neural_transfer(
        content_image_path=IMAGE_DIR / 'dancing.jpg',
        style_image_path=IMAGE_DIR / 'picasso.jpg'
    )