import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_training_history(history_path='training_history.json', save_path='training_curves.png'):
    """
    Reads training_history.json and generates a 2-panel training graph
    (Loss curve + Dice Coefficient curve) with dark-mode styling matching the reference.
    """
    if not os.path.exists(history_path):
        print(f"Error: '{history_path}' not found. Please run train.py first.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = list(range(1, len(history['train_loss']) + 1))
    n = len(epochs)

    # --- Styling to match reference image ---
    BG_COLOR   = '#0e1117'
    GRID_COLOR = '#2a2d35'
    TRAIN_CLR  = '#5ab4f5'
    VAL_CLR    = '#4ddb8a'

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.facecolor': BG_COLOR,
        'figure.facecolor': BG_COLOR,
        'axes.edgecolor': GRID_COLOR,
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': GRID_COLOR,
        'legend.facecolor': '#1c2030',
        'legend.edgecolor': GRID_COLOR,
        'text.color': 'white',
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_COLOR)

    marker = 'o' if n == 1 else None  # show dot if single point

    # ---- Panel 1: Loss Curves ----
    ax1.plot(epochs, history['train_loss'], color=TRAIN_CLR, linewidth=2.0,
             marker=marker, markersize=8, label='Train')
    ax1.plot(epochs, history['val_loss'],   color=VAL_CLR,   linewidth=2.0,
             linestyle='--', marker=marker, markersize=8, label='Validation')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold', color='white')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_xticks(epochs)                          # force integer ticks
    ax1.set_xlim(max(0.5, epochs[0] - 0.5), epochs[-1] + 0.5)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ---- Panel 2: Dice Coefficient Curves ----
    ax2.plot(epochs, history['train_dice'], color=TRAIN_CLR, linewidth=2.0,
             marker=marker, markersize=8, label='Train')
    ax2.plot(epochs, history['val_dice'],   color=VAL_CLR,   linewidth=2.0,
             linestyle='--', marker=marker, markersize=8, label='Validation')
    ax2.set_title('Training Dice Coefficient', fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Coefficient')
    ax2.set_xticks(epochs)                          # force integer ticks
    ax2.set_xlim(max(0.5, epochs[0] - 0.5), epochs[-1] + 0.5)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, facecolor=BG_COLOR, dpi=150)
    print(f"Training curves saved to '{save_path}'")

if __name__ == '__main__':
    plot_training_history()
