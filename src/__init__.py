
# Prevent transformers from importing torchvision (which can be missing or
# incompatible in many lightweight environments/containers).
import os as _os
_os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
_os.environ.setdefault("TRANSFORMERS_NO_LIBROSA", "1")

