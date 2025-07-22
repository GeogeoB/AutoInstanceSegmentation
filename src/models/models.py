from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from torchvision import transforms


class AutoInstanceSegmentation:
    class DinoV2ModelSize(Enum):
        SMALL = "dinov2_vits14_reg"
        BASE = "dinov2_vitb14_reg"
        LARGE = "dinov2_vitl14_reg"
        GIANT = "dinov2_vitg14_reg"

    def __init__(
        self,
        sam_ckpt_path: str,
        sam_model_cfg_path: str,
        dinov2_model_size: "AutoInstanceSegmentation.DinoV2ModelSize",
        device: str,
    ):
        self.device = device

        # Sam Configuration
        self.mask_generator = SAM2AutomaticMaskGenerator(
            build_sam2(sam_model_cfg_path, sam_ckpt_path, device=device)
        )

        # DinoV2 Configuration
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", dinov2_model_size.value)
        self.dinov2.eval()
        self.dinov2.to(self.device)

        self.dinov2_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                lambda x: 255.0 * x[:3],  # Discard alpha component and scale by 255
                transforms.Normalize(
                    mean=(123.675, 116.28, 103.53),
                    std=(58.395, 57.12, 57.375),
                ),
            ]
        )

    def _sam_prediction(self, image: np.array):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = self.mask_generator.generate(image)
            return [prediction["segmentation"] for prediction in predictions]

    def _dinov2_prediction(self, image: np.array):
        transformed_image = self.dinov2_transform(image)
        batch = transformed_image.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            return self.dinov2.get_intermediate_layers(batch, n=1)[0]

    def _compute_embeddings(self, image: np.array):
        embeddings = self._dinov2_prediction(image)
        embeddings = embeddings.view(1, 16, 16, 384).permute(0, 3, 1, 2)
        embeddings = (
            F.interpolate(
                embeddings, size=image.shape[:2], mode="bilinear", align_corners=False
            )
            .cpu()
            .numpy()
        )
        return embeddings

    def _compute_mask_embeddings(self, embeddings: np.array, mask: np.array):
        mask = mask[None, None, :, :]
        masked_embeddings = embeddings * mask
        sum_embeddings = masked_embeddings.sum(axis=(2, 3))
        num_valid = mask.sum()
        mean_embeddings = sum_embeddings / (num_valid + 1e-8)
        return mean_embeddings

    def add(self, image: np.array, mask: np.array, tag: str | None = None):
        embeddings = self._compute_embeddings(image)
        mean_embeddings = self._compute_mask_embeddings(embeddings, mask)

    def __call__(self, image: np.array):
        print("image shape", image.shape)
        masks = self._sam_prediction(image)
        embeddings = self._compute_embeddings(image)

        for mask in masks:
            mean_embeddings = self._compute_mask_embeddings(embeddings, mask)


if __name__ == "__main__":
    import os

    BASE_MODEL_DIR = os.getcwd()
    sam_ckpt_path = os.path.join(BASE_MODEL_DIR, "ckpt/sam2.1_hiera_small.pt")
    sam_model_cfg_path = "/" + os.path.join(
        BASE_MODEL_DIR, "ckpt/sam2_small_config.yaml"
    )
    dinov2_model_size = AutoInstanceSegmentation.DinoV2ModelSize.SMALL
    device = "cuda"

    auto_instance_segmentation = AutoInstanceSegmentation(
        sam_ckpt_path=sam_ckpt_path,
        sam_model_cfg_path=sam_model_cfg_path,
        dinov2_model_size=dinov2_model_size,
        device=device,
    )

    image = np.asarray(Image.open("test.png").convert("RGB"))
    auto_instance_segmentation(image)
