import numpy as np
from albumentations import (
    Compose, HorizontalFlip, Rotate, RandomBrightnessContrast,
    GaussianBlur, MotionBlur, RandomFog, RandomRain,
    RandomSunFlare, RandomShadow, OneOf
)

class AdvancedAugmentation:
    def __init__(self):
        self.transform = Compose([
            HorizontalFlip(p=0.5),
            Rotate(limit=15, p=0.3),
            OneOf([
                MotionBlur(blur_limit=7, p=1.0),
                GaussianBlur(blur_limit=7, p=1.0),
            ], p=0.4),
            OneOf([
                RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=1.0),
            ], p=0.3),
            OneOf([
                RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=1.0),
            ], p=0.4),
            RandomShadow(p=0.2),
        ])

    def apply_augmentation(self, image):
        if image.dtype == np.float32:
            img = (image*255).astype("uint8")
        else:
            img = image
        augmented = self.transform(image=img)["image"]
        return augmented.astype("float32")/255.0