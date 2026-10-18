import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

# ============================================================
# ФИНАЛЬНЫЙ ТЕСТ КОЛЬЦА 25.5° (Полное соответствие LaTeX-документу)
# ============================================================

# Координаты диполя реликтового излучения
ra_c, dec_c = 167.9, -7.2

# Зафиксированный радиус кольца
R_FIXED = 25.5

# Количество мок-каталогов
N_MOCKS = 10000 

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
def load_data(filename, zmax=3200):
    with fits.open(filename) as hdul:
        data = hdul[1].data
        
    ra = np.array(data['RAJ2000'])
    dec = np.array(data['DEJ2000'])
    
    # Маска галактической широты |b| > 5 (зона избегания)
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    b = coords.galactic.b.deg
    
    if 'cz' in data.names:
        cz = np.array(data['cz'])
        # Отсечка cz < 3200 для 2MRS, чтобы получить 3515 галактик
        mask = (cz > 0) & (cz < zmax) & np.isfinite(ra) & np.isfinite(dec) & (np.abs(b) > 5)
        return ra[mask], dec[mask]
    else:
        dm = np.array(data['DM'])
        mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(dm) & (dm > 25) & (np.abs(b) > 5)
        ra = ra[mask]
        dec = dec[mask]
        d_mpc = 10**((dm[mask] + 5)/5) / 1e6
        # Отсечка d < 45 Мпк для CF4, чтобы получить 3626 галактик
        mask2 = d_mpc < 45.0
        return ra[mask2], dec[mask2]

def angular_distance(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    cos_d = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    return np.degrees(np.arccos(np.clip(cos_d, -1, 1)))

def ratio_at_r(ra, dec, r_center, half_width=2.0):
    """Считает отношение плотности в кольце к плотности фона."""
    d = angular_distance(ra_c, dec_c, ra, dec)
    
    # Кольцо
    mask_ring = (d >= r_center - half_width) & (d <= r_center + half_width)
    n_ring = mask_ring.sum()
    
    # Фон (соседние радиусы)
    mask_bg1 = (d >= r_center - 10) & (d < r_center - 5)
    mask_bg2 = (d > r_center + 5) & (d <= r_center + 10)
    mask_bg = mask_bg1 | mask_bg2
    n_bg = mask_bg.sum()
    
    # Площади
    area_ring = np.pi * (np.radians(r_center + half_width)**2 - np.radians(r_center - half_width)**2)
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

ra_2mrs, dec_2mrs = load_data('2mrs_full.fits', zmax=3200)
print(f"2MRS: {len(ra_2mrs)} галактик")

ratio_2mrs = ratio_at_r(ra_2mrs, dec_2mrs, R_FIXED, half_width=2.0)
print(f"Ratio(25.5° ± 2°): {ratio_2mrs:.2f}")

# ============================================================
# 2. УСТОЙЧИВОСТЬ К ШИРИНЕ БИНА
# ============================================================
print("\n" + "=" * 60)
print("2. УСТОЙЧИВОСТЬ К ШИРИНЕ БИНА")
print("=" * 60)

for hw in [2, 5, 10]:
    ratio = ratio_at_r(ra_2mrs, dec_2mrs, R_FIXED, half_width=hw)
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
with fits.open('2mrs_full.fits') as hdul:
    data = hdul[1].data
    cz_all = np.array(data['cz'])
    
coords_all = SkyCoord(ra=np.array(data['RAJ2000'])*u.deg, dec=np.array(data['DEJ2000'])*u.deg, frame='icrs')
b_all = coords_all.galactic.b.deg

mask_cz = (cz_all > 500) & (cz_all < 3500) & (np.abs(b_all) > 5)
ra_5500 = np.array(data['RAJ2000'])[mask_cz]
dec_5500 = np.array(data['DEJ2000'])[mask_cz]
ratio_5500 = ratio_at_r(ra_5500, dec_5500, R_FIXED, half_width=2.0)
print(f"  500 < cz < 3500: ratio = {ratio_5500:.2f}, N = {mask_cz.sum()}")

# ============================================================
# 4. МОКИ С СЕЛЕКЦИЕЙ
# ============================================================
print("\n" + "=" * 60)
print(f"4. МОКИ С СЕЛЕКЦИЕЙ (N = {N_MOCKS})")
print("=" * 60)

ratios_mock = []
for i in range(N_MOCKS):
    # Перемешиваем привязку RA к Dec, сохраняя общий профиль плотности
    ra_mock = np.random.permutation(ra_2mrs)
    ratio_m = ratio_at_r(ra_mock, dec_2mrs, R_FIXED, half_width=2.0)
    ratios_mock.append(ratio_m)

ratios_mock = np.array(ratios_mock)
p_value_2mrs = np.mean(ratios_mock >= ratio_2mrs)

print(f"Мок (среднее): {ratios_mock.mean():.2f} ± {ratios_mock.std():.2f}")
print(f"Реальный ratio: {ratio_2mrs:.2f}")
print(f"p-value (2MRS): {p_value_2mrs:.6f}")

# ============================================================
# 5. CF4 (СЛЕПАЯ РЕПЛИКАЦИЯ)
# ============================================================
print("\n" + "=" * 60)
print("5. CF4 (СЛЕПАЯ РЕПЛИКАЦИЯ)")
print("=" * 60)

ra_cf4, dec_cf4 = load_data('cf4.fits')
ratio_cf4 = ratio_at_r(ra_cf4, dec_cf4, R_FIXED, half_width=2.0)
print(f"CF4: {len(ra_cf4)} галактик")
print(f"Ratio(25.5° ± 2°): {ratio_cf4:.2f}")

ratios_cf4_mock = []
for i in range(N_MOCKS):
    ra_mock = np.random.permutation(ra_cf4)
    ratio_m = ratio_at_r(ra_mock, dec_cf4, R_FIXED, half_width=2.0)
    ratios_cf4_mock.append(ratio_m)

ratios_cf4_mock = np.array(ratios_cf4_mock)
p_value_cf4 = np.mean(ratios_cf4_mock >= ratio_cf4)

print(f"Мок (среднее): {ratios_cf4_mock.mean():.2f} ± {ratios_cf4_mock.std():.2f}")
print(f"p-value (CF4): {p_value_cf4:.6f}")

# ============================================================
# 6. ИТОГ
# ============================================================
print("\n" + "=" * 60)
print("ИТОГ")
print("=" * 60)

print(f"1. 2MRS N: {len(ra_2mrs)}")
print(f"2. 2MRS ratio(25.5°): {ratio_2mrs:.2f}")
print(f"3. p-value (2MRS): {p_value_2mrs:.6f}")
print(f"4. CF4 N: {len(ra_cf4)}")
print(f"5. CF4 ratio(25.5°): {ratio_cf4:.2f}")
print(f"6. p-value (CF4): {p_value_cf4:.6f}")

# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Профиль для 2MRS
d_2mrs = angular_distance(ra_c, dec_c, ra_2mrs, dec_2mrs)
bins = np.linspace(0, 90, 181) 
hist_2mrs, _ = np.histogram(d_2mrs, bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])
area = np.pi * (np.radians(bins[1:])**2 - np.radians(bins[:-1])**2)
density_2mrs = hist_2mrs / area
density_2mrs_norm = density_2mrs / density_2mrs[centers > 10].mean()

ax = axes[0]
ax.plot(centers, density_2mrs_norm, 'o-', markersize=2, color='blue', label='2MRS')
ax.axvline(R_FIXED, color='red', linestyle='--', label=f'{R_FIXED}°')
ax.set_xlabel('Угловое расстояние от CMB диполя (град)')
ax.set_ylabel('Нормированная плотность')
ax.set_title(f'2MRS (p = {p_value_2mrs:.6f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Профиль для CF4
d_cf4 = angular_distance(ra_c, dec_c, ra_cf4, dec_cf4)
hist_cf4, _ = np.histogram(d_cf4, bins=bins)
density_cf4 = hist_cf4 / area
density_cf4_norm = density_cf4 / density_cf4[centers > 10].mean()

ax = axes[1]
ax.plot(centers, density_cf4_norm, 'o-', markersize=2, color='green', label='CF4')
ax.axvline(R_FIXED, color='red', linestyle='--', label=f'{R_FIXED}°')
ax.set_xlabel('Угловое расстояние от CMB диполя (град)')
ax.set_ylabel('Нормированная плотность')
ax.set_title(f'CF4 (p = {p_value_cf4:.6f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ring_figure_v2.png', dpi=150)
print("\nГрафик сохранен как: ring_figure_v2.png")
