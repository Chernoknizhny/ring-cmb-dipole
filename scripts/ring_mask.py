# -*- coding: utf-8 -*-
"""
ring_mask.py — тест на живучесть кольцевого избытка 25.5°.
ИСПРАВЛЕННАЯ ВЕРСИЯ (с отсечкой cz, правильными моками и секторами).
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# ---------- Константы модели ----------
DIP_RA, DIP_DEC = 167.9, -7.2          # диполь СМВ, градусы
RING_R, RING_W  = 25.5, 2.0            # кольцо 25.5° ± 2°
BG1, BG2        = (15.0, 20.0), (30.0, 35.0)  # фоновые оболочки
VIRGO           = (187.7, 12.7)        # ядро скопления Девы (M87)
VIRGO_CZ        = (800.0, 1800.0)      # cz-окно Девы
FORNAX          = (54.6, -35.4)        # скопление Форнакса
FORNAX_CZ       = (900.0, 1900.0)
N_MOCK          = 5000                 # моков на тест
M_MAX           = 60                   # максимум азимутальной гармоники
RNG             = np.random.default_rng(42)

D2R = np.pi / 180.0

# Вектор диполя (глобальная переменная, которая была забыта в оригинале)
DIP = np.array([np.cos(DIP_DEC * D2R) * np.cos(DIP_RA * D2R),
                np.cos(DIP_DEC * D2R) * np.sin(DIP_RA * D2R),
                np.sin(DIP_DEC * D2R)])

# ---------- Сферическая геометрия ----------
def unit(ra, dec):
    ra_, dec_ = ra * D2R, dec * D2R
    return np.stack([np.cos(dec_) * np.cos(ra_),
                     np.cos(dec_) * np.sin(ra_),
                     np.sin(dec_)], axis=-1)

def ang_dist(u1, u2):
    x = np.clip(np.sum(u1 * u2, axis=-1), -1.0, 1.0)
    return np.arccos(x) / D2R

def basis_perp(d):
    """Ортонормированный базис (e1, e2) в плоскости, перпендикулярной d."""
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return e1, e2

def to_gal(u_eq):
    """Экватор -> галактические (l, b), градусы."""
    M = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                  [ 0.4941094279, -0.4448296300,  0.7469822445],
                  [-0.8676661490, -0.1980763734,  0.4559837762]])
    g = u_eq @ M.T
    b = np.arcsin(np.clip(g[..., 2], -1, 1)) / D2R
    l = np.degrees(np.arctan2(g[..., 1], g[..., 0])) % 360.0
    return l, b

def from_gal(l, b):
    M = np.array([[-0.0548755604,  0.4941094279, -0.8676661490],
                  [-0.8734370902, -0.4448296300, -0.1980763734],
                  [-0.4838350155,  0.7469822445,  0.4559837762]])
    l_, b_ = l * D2R, b * D2R
    g = np.stack([np.cos(b_) * np.cos(l_), np.cos(b_) * np.sin(l_), np.sin(b_)], axis=-1)
    return g @ M.T

# ---------- Маски ----------
def mask_sel(ua, cz, centers_cz, radius):
    """True — галактику исключить (попадает в конус маски по углу И cz)."""
    kill = np.zeros(len(ua), dtype=bool)
    for (ra_c, dec_c), (cz0, cz1) in centers_cz:
        in_cone = ang_dist(ua, unit(ra_c, dec_c)) < radius
        in_cz   = (cz >= cz0) & (cz <= cz1)
        kill |= in_cone & in_cz
    return kill

# ---------- Ratio ----------
def ratio_masked(theta, keep):
    ring = keep & (np.abs(theta - RING_R) <= RING_W)
    bg   = keep & (((theta >= BG1[0]) & (theta <= BG1[1])) |
                   ((theta >= BG2[0]) & (theta <= BG2[1])))
    w_r = 2.0 * RING_W
    w_b = (BG1[1]-BG1[0]) + (BG2[1]-BG2[0])
    s = np.sin(np.radians(RING_R - RING_W/2))
    s_b = 0.5*(np.sin(np.radians((BG1[0]+BG1[1])/2)) +
               np.sin(np.radians((BG2[0]+BG2[1])/2)))
    dens_r = ring.sum() / (2*np.pi*s*w_r)
    dens_b = bg.sum()   / (2*np.pi*s_b*w_b)
    return dens_r / dens_b if dens_b > 0 else np.nan, ring.sum(), bg.sum()

def mock_ratio(theta, b, keep, n_mock=N_MOCK, seed=RNG):
    rng = np.random.default_rng(seed)
    out = np.empty(n_mock)
    for i in range(n_mock):
        l_rand = rng.uniform(0, 360, len(theta))
        u_m = from_gal(l_rand, b)
        t_m = ang_dist(u_m, DIP)
        out[i], _, _ = ratio_masked(t_m, keep)
    return out

# ---------- Азимут и «бусины» ----------
def azimuth(u, d=None):
    if d is None: d = DIP
    e1, e2 = basis_perp(d)
    return np.degrees(np.arctan2(u @ e2, u @ e1)) % 360.0

def power_spectrum(phi, m_max=M_MAX):
    N = len(phi)
    ms = np.arange(2, m_max + 1)
    z = np.exp(1j * np.outer(ms, phi * D2R)).sum(axis=1)
    return ms, (np.abs(z) ** 2) / N

def rayleigh(phi):
    N = len(phi)
    R = np.abs(np.exp(1j * phi * D2R).mean())
    return R, np.exp(-N * R * R)

# ---------- Вспомогательное ----------
def cz_stats(cz_ring, cz_bg):
    ks = ks_2samp(cz_ring, cz_bg)
    med = np.median(cz_ring)
    boot = np.array([np.std(RNG.choice(cz_ring, len(cz_ring))) for _ in range(1000)])
    return dict(ks_stat=ks.statistic, ks_p=ks.pvalue, med_cz=med,
                std_cz=np.std(cz_ring), std_lo=np.percentile(boot, 16),
                std_hi=np.percentile(boot, 84))

def block(title):
    print("\n" + "=" * 62); print(title); print("=" * 62)

# ---------- Основной анализ ----------
def analyze(ra, dec, cz, dist, name, mask_fornax=False):
    ua = unit(ra, dec)
    theta = ang_dist(ua, DIP)
    b = to_gal(ua)[1]
    keep0 = np.ones(len(ua), dtype=bool)

    block(f"Анализ {name}: N = {len(ra)}")
    R0, n_r, n_b = ratio_masked(theta, keep0)
    mocks0 = mock_ratio(theta, b, keep0)
    print(f"[1] Ratio без маски: {R0:.2f} | моки: {mocks0.mean():.2f} ± {mocks0.std():.2f} "
          f"| p = {max((mocks0 > R0).mean(), 1/len(mocks0)):.2e}  (кольцо {n_r}, фон {n_b})")

    # Маски Девы (10°, 15°) и Форнакса
    extra = [("Форнакс 10°", [FORNAX], [FORNAX_CZ], 10.0)] if mask_fornax else []
    for label, centers, czwin, rad in (
            [("Дева 10°", [VIRGO], [VIRGO_CZ], 10.0),
             ("Дева 15°", [VIRGO], [VIRGO_CZ], 15.0)] + extra):
        kill = mask_sel(ua, cz, list(zip(centers, czwin)), rad)
        keep = ~kill
        Rm, _, _ = ratio_masked(theta, keep)
        
        # Правильный мок с маской в пространстве мока
        rng = np.random.default_rng(1000 + int(rad))
        mock_vals = []
        for i in range(2000):
            l_m = rng.uniform(0, 360, len(theta))
            u_m = from_gal(l_m, b)
            kill_m = mask_sel(u_m, cz, list(zip(centers, czwin)), rad)
            Rm_m, _, _ = ratio_masked(ang_dist(u_m, DIP), ~kill_m)
            mock_vals.append(Rm_m)
            
        mock_vals = np.array(mock_vals)
        print(f"[2] Ratio после маски {label}: {Rm:.2f} "
              f"(сигнал сохранён на {Rm/R0*100:.0f}% от исходного; "
              f"моки {mock_vals.mean():.2f} ± {mock_vals.std():.2f})")
        killed_in_ring = (kill & (np.abs(theta - RING_R) <= RING_W)).sum()
        print(f"    исключено из кольца: {killed_in_ring} галактик из {n_r}")

    # Глубина: KS и ширина пика
    ring_cz = cz[np.abs(theta - RING_R) <= RING_W]
    bg_cz   = cz[(theta >= BG1[0]) & (theta <= BG1[1]) | (theta >= BG2[0]) & (theta <= BG2[1])]
    st = cz_stats(ring_cz, bg_cz)
    block(f"Глубина ({name})")
    print(f"KS-тест кольцо vs фон: D = {st['ks_stat']:.3f}, p = {st['ks_p']:.2e}")
    print(f"Медиана cz кольца: {st['med_cz']:.0f} км/с; std = {st['std_cz']:.0f} "
          f"(bootstrap {st['std_lo']:.0f}–{st['std_hi']:.0f} км/с)")
    hist, edges = np.histogram(ring_cz, bins=12, range=(0, 3200))
    print("Пик cz-гистограммы кольца:", f"{edges[np.argmax(hist)]:.0f} км/с")

    # Азимутальный тест «бусин»
    phi_r = azimuth(ua[np.abs(theta - RING_R) <= RING_W])
    ms, P = power_spectrum(phi_r)
    
    # Правильный нуль: перемешивание азимутов кольца
    rng = np.random.default_rng(777)
    P_mock = np.empty((2000, len(ms)))
    for i in range(2000):
        P_mock[i] = power_spectrum(rng.permutation(phi_r))[1]
        
    P_mean, P_std = P_mock.mean(0), P_mock.std(0)
    Z = (P - P_mean) / np.where(P_std > 0, P_std, 1)
    i_max = int(np.argmax(Z))
    p_max = min(1.0, max((P_mock.max(1) >= P[i_max]).mean(), 1/2000) * len(ms))
    R_ray, p_ray = rayleigh(phi_r)
    
    block(f"Азимутальный тест «бусин» ({name})")
    print(f"Рэлей: R = {R_ray:.3f}, p = {p_ray:.2e}")
    print(f"Максимум мощности: m = {ms[i_max]}, {Z[i_max]:.1f}σ, "
          f"p(с поправкой на {len(ms)} гармоник) = {p_max:.2e}")
    
    # Направление концентрации и сектора
    phi_bar = (np.angle(np.exp(1j * phi_r * D2R).mean()) / D2R) % 360
    sect = np.histogram(phi_r, bins=np.arange(0, 361, 30))[0]
    print(f"Средний азимут концентрации: {phi_bar:.0f}°; сектора (30°): {sect}")
    print(f"Самый плотный сектор: {sect.max()/len(phi_r)*100:.0f}% кольца (равномерно = 8%)")
    
    print(f"Число бусин, ожидаемое из Блока 5.5 (T=1e5–1e6 K): m ~ 14–45")
    print("Вывод: " + ("азимутальная мода есть" if p_max < 0.05 else
                       "азимутальная структура НЕ обнаружена (гладкое кольцо)"))

    # График
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.arange(5, 90, 1.25)
    prof0, _ = np.histogram(theta, bins=bins)
    profm, _ = np.histogram(theta[mask_sel(ua, cz, [ (VIRGO, VIRGO_CZ) ], 10.0)], bins=bins)
    area = 2*np.pi*np.sin(np.radians(0.5*(bins[:-1]+bins[1:]))) * np.diff(bins)
    ax[0].plot(0.5*(bins[:-1]+bins[1:]), prof0/area/prof0[5:].sum()*prof0[5:].size, lw=2, label="без маски")
    ax[0].plot(0.5*(bins[:-1]+bins[1:]), profm/area/max(profm[5:].sum(),1)*max(prof0[5:].sum(),1), lw=2, label="маска Девы 10°")
    ax[0].axvspan(RING_R-RING_W, RING_R+RING_W, alpha=0.15, color="red")
    ax[0].set_xlabel("θ от диполя, град"); ax[0].set_ylabel("плотность (отн.)")
    ax[0].set_title(f"{name}: профиль до/после маски"); ax[0].legend()
    
    ax[1].plot(ms, P, "r-", lw=2, label="кольцо")
    ax[1].fill_between(ms, P_mean-P_std, P_mean+P_std, color="gray", alpha=0.4, label="моки ±1σ")
    ax[1].axhline(P_mock.mean()+3*P_mock.std(), color="k", ls="--", lw=1)
    ax[1].set_xlabel("m — азимутальная гармоника"); ax[1].set_ylabel("P(m)")
    ax[1].set_yscale("log"); ax[1].set_title("Азимутальный спектр (тест «бусин»)"); ax[1].legend()
    plt.tight_layout()
    out = f"ring_mask_{name}.png"
    plt.savefig(out, dpi=150); print(f"\nСохранен график: {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--catalog", default="CAT")
    p.add_argument("--col-ra", default="RA"); p.add_argument("--col-dec", default="DEC")
    p.add_argument("--col-cz", default="CZ"); p.add_argument("--col-dist", default="DIST")
    p.add_argument("--fornax", action="store_true")
    p.add_argument("--czmax", type=float, default=None)  # <--- Добавлено
    a = p.parse_args()
    
    import pandas as pd
    df = pd.read_csv(a.file)
    
    if a.czmax:  # <--- Добавлено
        df = df[df[a.col_cz] < a.czmax]
        
    ra, dec, cz = df[a.col_ra].values, df[a.col_dec].values, df[a.col_cz].values
    dist = df[a.col_dist].values if a.col_dist in df else None
    analyze(ra, dec, cz, dist, a.catalog, a.fornax)

if __name__ == "__main__":
    main()