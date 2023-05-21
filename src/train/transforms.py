"""Example of transforms for training."""

import albumentations
from albumentations.pytorch import ToTensorV2

transforms = albumentations.Compose(
    [
        albumentations.ToFloat(),
        albumentations.Flip(p=0.5),
        # albumentations.OneOf(
        #     [
        #         albumentations.ElasticTransform(alpha=1, sigma=20, alpha_affine=10),
        #         albumentations.GridDistortion(num_steps=6, distort_limit=0.1),
        #         albumentations.OpticalDistortion(distort_limit=0.05, shift_limit=0.05),
        #     ],
        #     p=0.2,
        # ),
        # albumentations.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=0.5),
        # albumentations.core.composition.PerChannel(
        #     albumentations.OneOf(
        #         [
        #             albumentations.MotionBlur(p=0.05),
        #             albumentations.MedianBlur(blur_limit=3, p=0.05),
        #             albumentations.Blur(blur_limit=3, p=0.05),
        #         ]
        #     ),
        #     p=1.0,
        # ),
        albumentations.OneOf(
            [
                albumentations.CoarseDropout(
                    max_holes=16, max_height=128 // 16, max_width=256 // 16, fill_value=0, p=0.5
                ),
                albumentations.GridDropout(ratio=0.09, p=0.5),
            ],
            p=0.5,
        ),
        albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
        ToTensorV2(),
    ],
    additional_targets={
        "image": "image",
        "depth_image": "image",
        "label": "image",
    },
)
