import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ============================================================
# ПОЛНЫЙ ТЕСТ КОЛЬЦА
# ============================================================

ra_c, dec_c = 167.9, -7.2
R_FIXED = 25.5

def load_data(filename, zmax=3000):
    with fits.open(filename) as hdul:
        data = hdul[1].data
    ra = np.array(data['RAJ2000'])
    dec = np.array(data['DEJ2000'])
    if 'cz' in data.names:
        cz = np.array(data['cz'])
        mask = (cz > 0) & (cz < zmax) & np.isfinite(ra) & np.isfinite(dec)
    else:
        dm = np.array(data['DM'])
        mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(dm) & (dm > 25)
        ra = ra[mask]
        dec = dec[mask]
        d_mpc = 10**((dm[mask] + 5)/5) / 1e6
        mask2 = d_mpc < 100
        return ra[mask2], dec[mask2]
    return ra[mask], dec[mask]

def angular_distance(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    cos_d = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    return np.degrees(np.arccos(np.clip(cos_d, -1, 1)))

def ratio_at_r(ra, dec, r_center, half_width=2.0):
    """Ratio в кольце r_center ± half_width."""
    d = angular_distance(ra_c, dec_c, ra, dec)
    
    # Кольцо
    mask_ring = (d >= r_center - half_width) & (d <= r_center + half_width)
    n_ring = mask_ring.sum()
    
    # Соседние радиусы (15-20 и 30-35)
    mask_bg1 = (d >= r_center - 10) & (d < r_center - 5)
    mask_bg2 = (d > r_center + 5) & (d <= r_center + 10)
    mask_bg = mask_bg1 | mask_bg2
    n_bg = mask_bg.sum()
    
    # Площади
    area_ring = np.pi * (np.radians(r_center + half_width)**2 - 
                         np.radians(r_center - half_width)**2)
    area_bg = (np.pi * (np.radians(r_center - 5)**2 - np.radians(r_center - 10)**2) +
               np.pi * (np.radians(r_center + 10)**2 - np.radians(r_center + 5)**2))
    
    density_ring = n_ring / area_ring
    density_bg = n_bg / area_bg
    
    return density_ring / density_bg if density_bg > 0 else 0

# ============================================================
# 1. БАЗОВЫЙ ТЕСТ
# ============================================================

print("=" * 60)
print("1. БАЗОВЫЙ ТЕСТ")
print("=" * 60)

ra, dec = load_data('2mrs_full.fits')
print(f"2MRS: {len(ra)} галактик")

ratio_base = ratio_at_r(ra, dec, R_FIXED, half_width=2.0)
print(f"\nRatio(25.5° ± 2°): {ratio_base:.2f}")

# ============================================================
# 2. ШИРИНА БИНА
# ============================================================

print("\n" + "=" * 60)
print("2. УСТОЙЧИВОСТЬ К ШИРИНЕ БИНА")
print("=" * 60)

for hw in [2, 5, 10]:
    ratio = ratio_at_r(ra, dec, R_FIXED, half_width=hw)
    print(f"  ±{hw}°: ratio = {ratio:.2f}")

# ============================================================
# 3. УСТОЙЧИВОСТЬ К cz-ОБОЛОЧКЕ
# ============================================================

print("\n" + "=" * 60)
print("3. УСТОЙЧИВОСТЬ К cz-ОБОЛОЧКЕ")
print("=" * 60)

for zmax in [3000, 5000]:
    ra_z, dec_z = load_data('2mrs_full.fits', zmax=zmax)
    ratio = ratio_at_r(ra_z, dec_z, R_FIXED, half_width=2.0)
    print(f"  cz < {zmax}: ratio = {ratio:.2f}, N = {len(ra_z)}")

# 500-3500 (другая оболочка)
ra_z, dec_z = load_data('2mrs_full.fits', zmax=3500)
d_z = angular_distance(ra_c, dec_c, ra_z, dec_z)
# Оставляем только cz > 500
with fits.open('2mrs_full.fits') as hdul:
    data = hdul[1].data
cz_all = np.array(data['cz'])
mask_cz = (cz_all > 500) & (cz_all < 3500)
ra_5500 = np.array(data['RAJ2000'])[mask_cz]
dec_5500 = np.array(data['DEJ2000'])[mask_cz]
ratio_5500 = ratio_at_r(ra_5500, dec_5500, R_FIXED, half_width=2.0)
print(f"  500 < cz < 3500: ratio = {ratio_5500:.2f}, N = {mask_cz.sum()}")

# ============================================================
# 4. МОКИ С СЕЛЕКЦИЕЙ
# ============================================================

print("\n" + "=" * 60)
print("4. МОКИ С СЕЛЕКЦИЕЙ")
print("=" * 60)

# Сохраняем маску: перемешиваем RA только между реальными галактиками
n_mocks = 1000
ratios_mock = []

for i in range(n_mocks):
    ra_mock = np.random.permutation(ra)
    ratio_m = ratio_at_r(ra_mock, dec, R_FIXED, half_width=2.0)
    ratios_mock.append(ratio_m)

ratios_mock = np.array(ratios_mock)
p_value = np.mean(ratios_mock >= ratio_base)

print(f"\nМок (среднее): {ratios_mock.mean():.2f} ± {ratios_mock.std():.2f}")
print(f"Реальный ratio: {ratio_base:.2f}")
print(f"p-value: {p_value:.4f}")

# ============================================================
# 5. CF4 (слепая репликация)
# ============================================================

print("\n" + "=" * 60)
print("5. CF4 (слепая репликация)")
print("=" * 60)

ra_cf4, dec_cf4 = load_data('cf4.fits')
ratio_cf4 = ratio_at_r(ra_cf4, dec_cf4, R_FIXED, half_width=2.0)
print(f"CF4: {len(ra_cf4)} галактик")
print(f"Ratio(25.5° ± 2°): {ratio_cf4:.2f}")

# Моки для CF4
ratios_cf4_mock = []
for i in range(1000):
    ra_mock = np.random.permutation(ra_cf4)
    ratio_m = ratio_at_r(ra_mock, dec_cf4, R_FIXED, half_width=2.0)
    ratios_cf4_mock.append(ratio_m)

ratios_cf4_mock = np.array(ratios_cf4_mock)
p_value_cf4 = np.mean(ratios_cf4_mock >= ratio_cf4)

print(f"p-value (CF4): {p_value_cf4:.4f}")

# ============================================================
# ИТОГ
# ============================================================

print("\n" + "=" * 60)
print("ИТОГ")
print("=" * 60)

print(f"\n1. 2MRS ratio(25.5°): {ratio_base:.2f}")
print(f"2. p-value (2MRS): {p_value:.4f}")
print(f"3. CF4 ratio(25.5°): {ratio_cf4:.2f}")
print(f"4. p-value (CF4): {p_value_cf4:.4f}")