"""Print a compact table of run records: python code/peek.py [prefix]."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import record
pre = sys.argv[1] if len(sys.argv) > 1 else ""
for rid, r in record.load_records(pre).items():
    vs, vn = r.get("val_screen") or {}, r.get("val_native") or {}
    f = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"{rid:48s} {r['status']:9s} ep={r.get('selected_epoch','-')!s:>3} scr_mIoU={f(vs.get('miou'))} "
          f"nat_mIoU={f(vn.get('miou'))} fgDice={f(vn.get('fg_dice', vs.get('fg_dice')))} bf1t2={f(vn.get('bf1_t2_fg_mean', vs.get('bf1_t2_fg_mean')))} "
          f"vol={f(vn.get('vol_fg_dice'))} wall={r.get('wall_s',0)/60:.1f}m")
