"""Gate G0: the harness environment reproduces the tutorial (``docs/examples/segmentation.md``).

The tutorial's exact setting on its generated scenes (300 clauses per class, 3 x 3 patch, 8 position
bits, 8 epochs, seed 0; 220 training scenes, 80 held out) must reach test mIoU >= 0.90 (tutorial:
0.935). This is a check of the installation, not an experiment. Writes ``results/gate_G0.json``.
"""

import sys

import torch
import torchtsetlin as tt
from common import DEVICE, RESULTS, env_info, write_json

NEEDS, TUTORIAL = 0.90, 0.935


def main(device=DEVICE):
    tt.seed_everything(0)
    rgb, labels = tt.data.make_scenes(300, 32, seed=0)
    x = tt.data.ColorThermometerEncoder(n_bits=4, value_range=(0.0, 1.0))(rgb).float()
    model = tt.SegmentationTsetlinMachine(4, n_clauses=300, T=60, s=10.0, patch_size=3, position_encoding=8,
                                          weighted=True, n_states=100).to(device)
    trainer = tt.Trainer(model, device=device, batch_size=8)
    trainer.fit((x[:220], labels[:220]), epochs=8)
    model.eval()
    pred = torch.cat([model.predict(x[i:i + 16].to(device)) for i in range(220, 300, 16)]).cpu()
    miou = float(tt.metrics.mean_iou(pred, labels[220:], n_classes=4))
    ok = miou >= NEEDS
    write_json({"pass": ok, "test_mIoU": miou, "needs": NEEDS, "tutorial": TUTORIAL, "env": env_info(device)},
               RESULTS / "gate_G0.json")
    print(f"gate G0: test mIoU {miou:.3f} on the tutorial's scenes (needs {NEEDS}, tutorial {TUTORIAL}): "
          f"{'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
