import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

# ============================================================
# РИСУНОК КОЛЬЦА (v2)
# ============================================================

ra_c, dec_c = 167.9, -7.2
R_RING = 25.5

# 2MRS
with fits.open('2mrs_full.fits') as hdul:
    data = hdul[1].data
ra_2mrs = np.array(data['RAJ2000'])
dec_2mrs = np.array(data['DEJ2000'])
cz_2mrs = np.array(data['cz'])
mask = (cz_2mrs > 0) & (cz_2mrs < 3200) & np.isfinite(ra_2mrs) & np.isfinite(dec_2mrs)
ra_2mrs = ra_2mrs[mask]
dec_2mrs = dec_2mrs[mask]

# CF4
with fits.open('cf4.fits') as hdul:
    data = hdul[1].data
ra_cf4 = np.array(data['RAJ2000'])
dec_cf4 = np.array(data['DEJ2000'])
dm_cf4 = np.array(data['DM'])
mask = np.isfinite(ra_cf4) & np.isfinite(dec_cf4) & np.isfinite(dm_cf4) & (dm_cf4 > 25)
ra_cf4 = ra_cf4[mask]
dec_cf4 = dec_cf4[mask]
dm_cf4 = dm_cf4[mask]
d_cf4 = 10**((dm_cf4 + 5)/5) / 1e6
mask2 = d_cf4 < 45
ra_cf4 = ra_cf4[mask2]
dec_cf4 = dec_cf4[mask2]

def angular_distance(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    cos_d = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    return np.degrees(np.arccos(np.clip(cos_d, -1, 1)))

# ============================================================
# РИСУНОК
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# --- Панель 1: карта неба ---
ax = axes[0]

ax.scatter(ra_2mrs, dec_2mrs, s=1, c='gray', alpha=0.3, label='2MRS')

ax.scatter(ra_c, dec_c, c='yellow', s=300, marker='*', 
           edgecolors='black', linewidth=2, zorder=5, label='CMB dipole')

# Кольцо (эллипс из-за проекции)
theta = np.linspace(0, 2*np.pi, 200)
ring_ra = ra_c + R_RING * np.cos(theta) / np.cos(np.radians(dec_c))
ring_dec = dec_c + R_RING * np.sin(theta)
ax.plot(ring_ra, ring_dec, 'r-', linewidth=3, alpha=0.8, label=f'Ring {R_RING}°')

# Толщина кольца
for r in [R_RING - 2, R_RING + 2]:
    ring_ra = ra_c + r * np.cos(theta) / np.cos(np.radians(dec_c))
    ring_dec = dec_c + r * np.sin(theta)
    ax.plot(ring_ra, ring_dec, 'r--', linewidth=1, alpha=0.5)

ax.set_xlabel('RA (deg)', fontsize=14)
ax.set_ylabel('Dec (deg)', fontsize=14)
ax.set_title('2MRS: Ring at 25.5° around CMB dipole', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 360)
ax.set_ylim(-90, 90)

# Подпись
ax.text(0.02, 0.02, 'Note: ring is an ellipse due to RA/Dec projection', 
        transform=ax.transAxes, fontsize=10, style='italic', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- Панель 2: радиальный профиль ---
ax2 = axes[1]

# 2MRS
d_2mrs = angular_distance(ra_c, dec_c, ra_2mrs, dec_2mrs)
bins = np.linspace(0, 90, 91)  # шаг 1°
hist_2mrs, _ = np.histogram(d_2mrs, bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])
area = np.pi * (np.radians(bins[1:])**2 - np.radians(bins[:-1])**2)
density_2mrs = hist_2mrs / area
density_2mrs_norm = density_2mrs / density_2mrs[centers > 10].mean()

# Сглаживание
density_2mrs_smooth = gaussian_filter1d(density_2mrs_norm, sigma=1.5)

ax2.plot(centers, density_2mrs_norm, 'o', markersize=3, 
         color='blue', alpha=0.3, label='2MRS (raw)')
ax2.plot(centers, density_2mrs_smooth, '-', linewidth=2, 
         color='blue', label='2MRS (smoothed)')

# CF4
d_cf4 = angular_distance(ra_c, dec_c, ra_cf4, dec_cf4)
hist_cf4, _ = np.histogram(d_cf4, bins=bins)
density_cf4 = hist_cf4 / area
density_cf4_norm = density_cf4 / density_cf4[centers > 10].mean()
density_cf4_smooth = gaussian_filter1d(density_cf4_norm, sigma=1.5)

ax2.plot(centers, density_cf4_smooth, '-', linewidth=2, 
         color='green', label='CF4 (smoothed)')

ax2.axvline(R_RING, color='red', linestyle='--', linewidth=2, 
            label=f'{R_RING}°')
ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Angular distance from CMB dipole (deg)', fontsize=14)
ax2.set_ylabel('Normalized density', fontsize=14)
ax2.set_title('Radial profile: 2MRS vs CF4', fontsize=16, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 60)

plt.tight_layout()
plt.savefig('ring_figure_v2.png', dpi=300, bbox_inches='tight')
print("Сохранено: ring_figure_v2.png")