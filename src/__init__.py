import sys

import torch

IS_COLAB = "google.colab" in sys.modules
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
__version__ = "0.0.1"
__all__ = ["IS_COLAB", "DEVICE"]
