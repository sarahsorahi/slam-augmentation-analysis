# =====================================
# CLEAN ACL-STYLE PLOTS
# =====================================

import matplotlib.pyplot as plt
import numpy as np

# Short labels for paper
short_labels = ["Real", "Near", "Middle", "Far", "Random", "Balanced"]
x = np.arange(len(short_labels))

# ---------- Macro-F1 ----------
plt.figure(figsize=(6,4))

plt.bar(
    x,
    agg["mean_macro_f1"],
    yerr=agg["std_macro_f1"],
    capsize=4
)

plt.xticks(x, short_labels)
plt.ylabel("Macro-F1")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(BASE_DIR / "figure_macro_f1_clean.png", dpi=300)
plt.show()


# ---------- Macro-AUC ----------
plt.figure(figsize=(6,4))

plt.bar(
    x,
    agg["mean_macro_auc"],
    yerr=agg["std_macro_auc"],
    capsize=4
)

plt.xticks(x, short_labels)
plt.ylabel("Macro-AUC")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(BASE_DIR / "figure_macro_auc_clean.png", dpi=300)
plt.show()