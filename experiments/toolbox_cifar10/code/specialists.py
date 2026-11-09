"""The 22 TM specialists of the Optimized Toolbox (arXiv:2406.00704): hyper-parameters from
Table III, published accuracies from Table IV (mean ± sample variance over the last 25 of 250
epochs, CIFAR-10 test set), clause counts per class.

Table III is used where the authors' scripts disagree with it (README.md, M2): the adaptive
Gaussian scripts default to 10x10 / T=1500 / s=10 and the 5x5 augmented colour thermometer script
to patch_size=4; Table III says 5x5 / T=2500 / s=3.5 and 5x5.
"""

CLAUSE_COLUMNS = (2000, 4000, 8000, 16000, 32000, 64000)
BEST_CLAUSES = 64000                    # the paper's best column
MAX_INCLUDED_LITERALS = 32              # every script
EPOCHS = 250                            # Table III
LAST_EPOCHS = 25                        # Table IV statistic

# base id: (name, encoding, model, window, weighted, T, s)
_BASE = {
    "canny10": ("10x10 Canny Edge Detection", "canny", "conv", 10, True, 1500, 10.0),
    "agauss5": ("5x5 Adaptive Gaussian Thresholding", "agauss", "conv", 5, True, 2500, 3.5),
    "amean10": ("10x10 Adaptive Mean Thresholding", "amean", "conv", 10, True, 500, 10.0),
    "otsu10": ("10x10 Otsu's Thresholding", "otsu", "conv", 10, True, 3000, 10.0),
    "therm3": ("3x3 Color Thermometers", "therm", "conv", 3, True, 3000, 5.0),
    "therm4": ("4x4 Color Thermometers", "therm", "conv", 4, True, 3000, 5.0),
    "therm5": ("5x5 Color Thermometers", "therm", "conv", 5, True, 3000, 5.0),
    "hog": ("Histogram of Oriented Gradients", "hog", "flat", 32, False, 50, 10.0),
    "atherm3": ("3x3 Adaptive Color Thermometers", "atherm", "conv", 3, True, 3000, 5.0),
    "atherm4": ("4x4 Adaptive Color Thermometers", "atherm", "conv", 4, True, 3000, 5.0),
    "atherm5": ("5x5 Adaptive Color Thermometers", "atherm", "conv", 5, True, 3000, 5.0),
}

# Table IV, (mean, sample variance) per clause column 2k..64k; key = specialist id.
TABLE_IV = {
    "canny10": [(48.7, .22), (50.4, .13), (53.4, .11), (56.6, .06), (58.8, .06), (60.6, .03)],
    "canny10_aug": [(50.6, .20), (51.5, .21), (53.2, .29), (56.5, .21), (59.9, .09), (62.4, .06)],
    "agauss5": [(57.9, .16), (60.6, .17), (63.0, .11), (66.7, .10), (68.2, .03), (70.2, .03)],
    "agauss5_aug": [(57.8, .24), (60.3, .15), (63.3, .11), (65.9, .04), (69.0, .04), (70.3, .02)],
    "amean10": [(57.8, .14), (60.3, .12), (62.3, .05), (65.4, .08), (67.2, .06), (68.4, .13)],
    "amean10_aug": [(59.1, .20), (61.4, .16), (63.4, .18), (65.7, .06), (68.3, .06), (70.2, .06)],
    "otsu10": [(56.9, .13), (58.5, .19), (61.2, .17), (62.9, .12), (63.9, .23), (65.0, .20)],
    "otsu10_aug": [(58.5, .17), (60.1, .21), (61.5, .23), (63.2, .20), (65.2, .19), (65.7, .31)],
    "therm3": [(63.6, .15), (65.8, .42), (68.4, .35), (70.4, .26), (72.3, .22), (74.3, .15)],
    "therm4": [(64.6, .16), (66.6, .33), (68.9, .44), (71.5, .30), (73.6, .31), (75.1, .32)],
    "therm5": [(64.5, .20), (66.8, .17), (69.1, .63), (71.7, .27), (73.7, .22), (75.4, .09)],
    "therm3_aug": [(64.1, .19), (65.4, .52), (67.5, .86), (69.1, 1.26), (70.5, 1.86), (72.7, .96)],
    "therm4_aug": [(65.4, .22), (66.4, .46), (68.5, .52), (69.7, 1.49), (71.6, 1.07), (73.6, 1.44)],
    "therm5_aug": [(65.1, .48), (66.9, .46), (68.6, .81), (69.8, 1.87), (71.7, 2.15), (74.1, 1.35)],
    "hog": [(64.2, .06), (65.7, .04), (66.9, .04), (67.3, .03), (67.5, .02), (67.5, .02)],
    "hog_aug": [(64.8, .07), (66.6, .08), (67.5, .03), (67.9, .02), (68.4, .03), (68.5, .02)],
    "atherm3": [(60.8, .10), (63.2, .30), (65.3, .14), (68.1, .05), (70.3, .06), (71.7, .06)],
    "atherm4": [(61.3, .30), (64.2, .22), (66.9, .25), (69.6, .07), (71.5, .07), (73.0, .05)],
    "atherm5": [(62.0, .25), (65.1, .16), (67.2, .14), (69.3, .06), (71.6, .06), (73.3, .04)],
    "atherm3_aug": [(61.3, .25), (62.6, .22), (64.8, .26), (67.1, .53), (68.9, .45), (70.8, .65)],
    "atherm4_aug": [(62.6, .23), (64.1, .36), (66.2, .21), (68.4, .33), (70.3, .75), (72.9, .33)],
    "atherm5_aug": [(62.9, .21), (64.6, .34), (66.4, .39), (68.5, .65), (71.1, .40), (72.9, .38)],
}
COMPOSITE_TABLE_IV = [(79.5, .04), (80.6, .02), (81.5, .02), (82.2, .02), (82.7, .02), (82.8, .01)]

# Table IV row order
ORDER = ["canny10", "canny10_aug", "agauss5", "agauss5_aug", "amean10", "amean10_aug", "otsu10",
         "otsu10_aug", "therm3", "therm4", "therm5", "therm3_aug", "therm4_aug", "therm5_aug", "hog",
         "hog_aug", "atherm3", "atherm4", "atherm5", "atherm3_aug", "atherm4_aug", "atherm5_aug"]


def _make():
    out = {}
    for sid in ORDER:
        base, aug = sid.removesuffix("_aug"), sid.endswith("_aug")
        name, enc, model, window, weighted, T, s = _BASE[base]
        if aug:
            name = name.replace(name.split(" ", 1)[1], "Augmented " + name.split(" ", 1)[1]) \
                if name[0].isdigit() else "Augmented " + name
        out[sid] = dict(id=sid, name=name, encoding=enc, augmented=aug, model=model,
                        window=window, weighted=weighted, T=T, s=s,
                        paper={c: {"mean": m, "variance": v}
                               for c, (m, v) in zip(CLAUSE_COLUMNS, TABLE_IV[sid])})
    return out


SPECIALISTS = _make()
assert len(SPECIALISTS) == 22
