from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from scipy.spatial.distance import cosine
from torchvision import transforms


@dataclass
class MemoryItem:
    embedding: torch.Tensor
    tag: Any


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
        score_treshold: float,
        device: str,
    ):
        self.device = device
        self.score_treshold = score_treshold

        # Sam Configuration
        self.mask_generator = SAM2AutomaticMaskGenerator(
            build_sam2(
                sam_model_cfg_path,
                sam_ckpt_path,
                device=device,
                apply_postprocessing=True,
            ),
        )

        # DinoV2 Configuration
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", dinov2_model_size.value)
        self.dinov2.eval()
        self.dinov2.to(self.device)

        self.dinov2_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((518, 518)),
                lambda x: 255.0 * x[:3],  # Discard alpha component and scale by 255
                transforms.Normalize(
                    mean=(123.675, 116.28, 103.53),
                    std=(58.395, 57.12, 57.375),
                ),
            ]
        )

        self.memory = []

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
        embeddings = embeddings.view(1, 37, 37, 384).permute(0, 3, 1, 2)
        embeddings = (
            F.interpolate(embeddings, size=image.shape[:2], mode="nearest")
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
        mean_embedding = self._compute_mask_embeddings(embeddings, mask)

        self.memory.append(MemoryItem(embedding=mean_embedding, tag=tag))

    def find_closest_and_get_score(self, embedding: np.array):
        closest = None
        score = np.inf
        for item in self.memory:
            memory_embedding = item.embedding

            item_score = cosine(memory_embedding.squeeze(), embedding.squeeze())
            if item_score < score:
                score = item_score
                closest = item

        return closest, score

    def __call__(self, image: np.array):
        masks = self._sam_prediction(image)
        embeddings = self._compute_embeddings(image)

        results = []
        for mask in masks:
            mean_embeddings = self._compute_mask_embeddings(
                embeddings, mask.astype(np.uint8)
            )
            closest, score = self.find_closest_and_get_score(mean_embeddings)

            if closest and score < 1 - self.score_treshold:
                results.append([mask, score, closest.tag])

        return results
