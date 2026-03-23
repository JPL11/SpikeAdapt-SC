"""Merge fig2a (AID masks) and fig2b (RESISC45 masks) into a single figure."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Load existing mask images
aid_img = mpimg.imread('paper/figures/fig2_mask_diversity_aid.png')
res_img = mpimg.imread('paper/figures/fig2_mask_diversity_resisc45.png')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.0))

ax1.imshow(aid_img)
ax1.set_title('(a) AID (30 classes)', fontweight='bold', fontsize=10)
ax1.axis('off')

ax2.imshow(res_img)
ax2.set_title('(b) RESISC45 (45 classes)', fontweight='bold', fontsize=10)
ax2.axis('off')

plt.tight_layout()
plt.savefig('paper/figures/fig2_mask_diversity_merged.png')
plt.savefig('paper/figures/fig2_mask_diversity_merged.pdf')
plt.close()
print("✅ Merged fig2 saved")
