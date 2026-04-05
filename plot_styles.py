import matplotlib.pyplot as plt
import os

def set_pub_style():
    """Configures Matplotlib for professional publication-quality plots."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1
    })

def get_pub_colors():
    """Returns a professional color palette."""
    return {
        "real": "#2E4053",      # Dark blue-gray
        "pred": "#E67E22",      # Deep orange
        "train_loss": "#2980B9", # Bright blue
        "val_loss": "#C0392B"    # Deep red
    }

def save_publication_figures(name, path_prefix="outputs/"):
    """Saves the current figure in multiple high-quality formats."""
    os.makedirs(path_prefix, exist_ok=True)
    formats = ["png", "pdf", "svg"]
    for fmt in formats:
        plt.savefig(os.path.join(path_prefix, f"{name}.{fmt}"), dpi=300)
    print(f"   - Gráficos guardados: {name}.[png, pdf, svg] en {path_prefix}")
