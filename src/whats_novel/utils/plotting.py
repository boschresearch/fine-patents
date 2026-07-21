# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: MIT

# region latex template
# LATEX snippet to get font sizes

# % \makeatletter

# % {\Huge "Huge": \number\f@size,}

# % {\huge "huge": \number\f@size,}

# % {\LARGE "LARGE": \number\f@size,}

# % {\Large "Large": \number\f@size,}

# % {\large "large": \number\f@size,}

# % {\normalsize "normalsize": \number\f@size,}

# % {\small "small": \number\f@size,}

# % {\footnotesize "footnotesize": \number\f@size,}

# % {\scriptsize "scriptsize": \number\f@size,}

# % {\tiny "tiny": \number\f@size,}

# % \makeatother
# endregion

import matplotlib.figure
import matplotlib.axes
from matplotlib.backends import backend_pgf as mpl_backend_pgf
from matplotlib import backend_bases as mpl_backend_bases
from matplotlib import pyplot as plt

latex_settings: dict = dict(
    font_sizes={
        "tiny": 6,
        "scriptsize": 8,
        "footnotesize": 10,
        "small": 10.95,
        "normalsize": 12,
        "large": 14.4,
        "Large": 17.28,
        "LARGE": 20.74,
        "huge": 24.88,
        "Huge": 24.88,
    },
    textwidth=415.55249,
    pt_per_inch=72.27,
)

mpl_backend_bases.register_backend("pdf", mpl_backend_pgf.FigureCanvasPgf)

plt.rcdefaults()
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": latex_settings["font_sizes"]["small"],
        "axes.labelsize": latex_settings["font_sizes"]["small"],
        "axes.titlesize": latex_settings["font_sizes"]["small"],
        "xtick.labelsize": latex_settings["font_sizes"]["scriptsize"],
        "ytick.labelsize": latex_settings["font_sizes"]["scriptsize"],
        "legend.fontsize": latex_settings["font_sizes"]["small"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
    }
)


def subplots(
    width: float, aspect_ratio: float = 4, **kwargs
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    width_in = width * latex_settings["textwidth"] / latex_settings["pt_per_inch"]
    height_in = width_in / aspect_ratio
    if "figsize" in kwargs:
        print("figsize not supported directly. Use width and aspect ratio")
        del kwargs["figsize"]
    return plt.subplots(figsize=(width_in, height_in), **kwargs)


def savefig(fig, path):
    fig.savefig(path, backend="pgf", bbox_inches="tight")
