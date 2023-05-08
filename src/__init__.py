import sys

import torch

IS_COLAB = "google.colab" in sys.modules
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

__version__ = "0.0.0"
__all__ = ["IS_COLAB", "DEVICE"]
