import os
import sys
from pathlib import Path

os.environ.setdefault("SS_WANDB", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
