import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ============================================================
# ФИНАЛЬНЫЙ ТЕСТ КОЛЬЦА 25.5°
# ============================================================

# CMB диполь (правильный знак!)
ra_c, dec_c = 167.9, -7.2

# Зафиксированный радиус
R_FIXED = 25.5

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_data(filename, zmax=3000):
    with fits.open(filename) as hdul:
        data = hdul[1].data
    
    ra = np.array(data['RAJ2000'])
    dec = np.array(data['DEJ2000'])
    
    if 'cz' in data.names:
        cz = np.array(data['cz'])
        mask = (cz > 0) & (cz < zmax) & np.isfinite(ra) & np.isfinite(dec)
        ra = ra[mask]
        dec = dec[mask]
    else:
        dm = np.array(data['DM'])
        mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(dm) & (dm > 25)
        ra = ra[mask]
        dec = dec[mask]
        d_mpc = 10**((dm[mask] + 5)/5) / 1e6
        mask2 = d_mpc < 100
        ra = ra[mask2]
        dec = dec[mask2]
    
    return ra, dec

def angular_distance(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    cos_d = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    return np.degrees(np.arccos(np.clip(cos_d, -1, 1)))

# ============================================================
# 1. 2MRS (основной каталог)
# ============================================================

print("=" * 60)
print("1. 2MRS")
print("=" * 60)

ra_2mrs, dec_2mrs = load_data('2mrs_full.fits')
print(f"Галактик: {len(ra_2mrs)}")

# Радиальный профиль
d_2mrs = angular_distance(ra_c, dec_c, ra_2mrs, dec_2mrs)

bins = np.linspace(0, 90, 181)  # шаг 0.5°
hist_2mrs, _ = np.histogram(d_2mrs, bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])
area = np.pi * (np.radians(bins[1:])**2 - np.radians(bins[:-1])**2)
density_2mrs = hist_2mrs / area
density_2mrs_norm = density_2mrs / density_2mrs[centers > 10].mean()

# Пик на 25.5°
idx_25 = np.argmin(np.abs(centers - R_FIXED))
ratio_25_2mrs = density_2mrs_norm[idx_25]
print(f"Ratio(25.5°): {ratio_25_2mrs:.2f}")

# ============================================================
# 2. МОКИ: РАНДОМИЗАЦИЯ АЗИМУТА
# ============================================================

print("\n" + "=" * 60)
print("2. МОКИ (рандомизация азимута)")
print("=" * 60)

n_mocks = 10000
ratios_mock = []
max_ratios_mock = []

print(f"Генерирую {n_mocks} моков...")

for i in range(n_mocks):
    if i % 1000 == 0:
        print(f"  {i}/{n_mocks}")
    
    # Рандомизация азимута: сохраняем d_2mrs, меняем RA
    ra_mock = np.random.uniform(0, 360, len(ra_2mrs))
    
    # Для каждого мока - пересчитываем расстояние от CMB
    d_mock = angular_distance(ra_c, dec_c, ra_mock, dec_2mrs)
    
    hist_mock, _ = np.histogram(d_mock, bins=bins)
    density_mock = hist_mock / area
    density_mock_norm = density_mock / density_mock[centers > 10].mean()
    
    ratios_mock.append(density_mock_norm[idx_25])
    max_ratios_mock.append(np.max(density_mock_norm[centers > 10]))

ratios_mock = np.array(ratios_mock)
max_ratios_mock = np.array(max_ratios_mock)

# p-value на 25.5°
n_greater = np.sum(ratios_mock >= ratio_25_2mrs)
p_value_25 = (n_greater + 1) / (n_mocks + 1)

# p-value с поправкой на множественность (max по профилю)
p_value_max = np.mean(max_ratios_mock >= ratio_25_2mrs)

print(f"\nРеальный ratio(25.5°): {ratio_25_2mrs:.2f}")
print(f"Мок (среднее): {ratios_mock.mean():.2f} ± {ratios_mock.std():.2f}")
print(f"p-value (25.5°): {p_value_25:.4f}")
print(f"p-value (max по профилю): {p_value_max:.4f}")

if p_value_max < 0.05:
    print("→ Кольцо ЗНАЧИМО даже с поправкой")
else:
    print("→ Кольцо НЕ значимо после поправки")

# ============================================================
# 3. СЛЕПАЯ РЕПЛИКАЦИЯ CF4
# ============================================================

print("\n" + "=" * 60)
print("3. СЛЕПАЯ РЕПЛИКАЦИЯ CF4")
print("=" * 60)

ra_cf4, dec_cf4 = load_data('cf4.fits')
print(f"Галактик: {len(ra_cf4)}")

d_cf4 = angular_distance(ra_c, dec_c, ra_cf4, dec_cf4)

hist_cf4, _ = np.histogram(d_cf4, bins=bins)
density_cf4 = hist_cf4 / area
density_cf4_norm = density_cf4 / density_cf4[centers > 10].mean()

ratio_25_cf4 = density_cf4_norm[idx_25]
print(f"Ratio(25.5°): {ratio_25_cf4:.2f}")

# Мок для CF4
ratios_cf4_mock = []

for i in range(1000):  # 1000 моков достаточно для репликации
    ra_mock = np.random.uniform(0, 360, len(ra_cf4))
    d_mock = angular_distance(ra_c, dec_c, ra_mock, dec_cf4)
    
    hist_mock, _ = np.histogram(d_mock, bins=bins)
    density_mock = hist_mock / area
    density_mock_norm = density_mock / density_mock[centers > 10].mean()
    
    ratios_cf4_mock.append(density_mock_norm[idx_25])

ratios_cf4_mock = np.array(ratios_cf4_mock)
n_greater_cf4 = np.sum(ratios_cf4_mock >= ratio_25_cf4)
p_value_cf4 = (n_greater_cf4 + 1) / (1001)

print(f"p-value (CF4): {p_value_cf4:.4f}")

if p_value_cf4 < 0.05:
    print("→ CF4 ПОДТВЕРЖДАЕТ кольцо")
else:
    print("→ CF4 НЕ подтверждает")

# ============================================================
# 4. ИТОГ
# ============================================================

print("\n" + "=" * 60)
print("ИТОГ")
print("=" * 60)

print(f"\n2MRS:")
print(f"  ratio(25.5°) = {ratio_25_2mrs:.2f}")
print(f"  p-value (max) = {p_value_max:.4f}")

print(f"\nCF4 (слепая репликация):")
print(f"  ratio(25.5°) = {ratio_25_cf4:.2f}")
print(f"  p-value = {p_value_cf4:.4f}")

# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.plot(centers, density_2mrs_norm, 'o-', markersize=2, label='2MRS')
ax.axvline(R_FIXED, color='red', linestyle='--', label=f'{R_FIXED}°')
ax.set_xlabel('r (deg)')
ax.set_ylabel('Нормированная плотность')
ax.set_title(f'2MRS: p(max) = {p_value_max:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(centers, density_cf4_norm, 'o-', markersize=2, label='CF4', color='green')
ax.axvline(R_FIXED, color='red', linestyle='--', label=f'{R_FIXED}°')
ax.set_xlabel('r (deg)')
ax.set_ylabel('Нормированная плотность')
ax.set_title(f'CF4: p = {p_value_cf4:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ring_final.png', dpi=150)
print("\nСохранено: ring_final.png")