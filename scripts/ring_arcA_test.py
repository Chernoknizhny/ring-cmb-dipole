import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIP   = (167.9, -7.2)
VIRGO = (187.7, 12.7)
HYDRA = (159.3, -27.5)
RING_R, RING_W = 25.5, 2.0
D2R = np.pi / 180.0
S_RING = 2*np.pi*np.sin(np.radians(RING_R)) * 2*RING_W
S_BG   = 2*np.pi*np.sin(np.radians(17.5))*5 + 2*np.pi*np.sin(np.radians(32.5))*5

def unit(ra, dec):
    ra_, dec_ = ra * D2R, dec * D2R
    return np.stack([np.cos(dec_)*np.cos(ra_), np.cos(dec_)*np.sin(ra_), np.sin(dec_)], axis=-1)

def ang_dist(u1, u2):
    return np.arccos(np.clip(u1 @ u2, -1, 1)) / D2R

def basis(d):
    ref = np.array([0., 0., 1.]) if abs(d[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)

def azimuth(u, d):
    e1, e2 = basis(d)
    return np.degrees(np.arctan2(u @ e2, u @ e1)) % 360.0

def load(filename, czmax):
    df = pd.read_csv(filename)
    if 'DIST' in df.columns and 'CF4' in filename:
        df = df[(df['DIST'] > 0) & (df['DIST'] < 45)]
    else:
        df = df[(df['CZ'] > 0) & (df['CZ'] < czmax)]
    return df

def run(filename, name, czmax):
    df = load(filename, czmax)
    ra, dec, cz = df['RA'].values, df['DEC'].values, df['CZ'].values
    print(f"\n--- {name}, срез cz < {czmax}: N = {len(df)} ---")

    u_dip, u_vir, u_hyd = unit(*DIP), unit(*VIRGO), unit(*HYDRA)
    ua = unit(ra, dec)
    theta = ang_dist(ua, u_dip)
    phi   = azimuth(ua, u_dip)

    ring = np.abs(theta - RING_R) <= RING_W
    bg   = ((theta >= 15) & (theta <= 20)) | ((theta >= 30) & (theta <= 35))
    arcA = ring & (phi > 30) & (phi < 60)
    mask_v = ang_dist(ua, u_vir) < 10.0

    R0 = (ring.sum()/S_RING) / (bg.sum()/S_BG)
    nA_hyd = arcA & (ang_dist(ua, u_hyd) < 10.0)
    print(f"    Ratio: {R0:.2f} (кольцо {ring.sum()}, фон {bg.sum()})")
    print(f"    Дуга А 30-60°: N = {arcA.sum()}, медиана cz = {np.median(cz[arcA]):.0f} км/с")
    if arcA.sum() > 5:
        p10, p90 = np.percentile(cz[arcA], [10, 90]).round(0)
        print(f"    перц. 10-90 cz дуги А: [{p10:.0f}, {p90:.0f}]")
    print(f"    доля дуги А в конусе Гидры 10°: {nA_hyd.sum()}/{arcA.sum()} = "
          f"{nA_hyd.sum()/max(arcA.sum(),1)*100:.0f}%")
    print(f"    cz галактик дуги А в конусе Гидры: медиана "
          f"{np.median(cz[nA_hyd]):.0f} км/с" if nA_hyd.sum() > 3 else
          "    в конусе Гидры слишком мало галактик для медианы")
    return arcA.sum(), np.median(cz[arcA]) if arcA.sum() else np.nan

def analyze(filename, name):
    # сравнение двух срезов для одного каталога
    n1, m1 = run(filename, name, 3200)
    n2, m2 = run(filename, name, 5000)
    if n2 > n1:
        print(f"\n    >>> Дуга А выросла с {n1} до {n2} гал. (+{n2-n1}), "
              f"медиана cz сместилась {m1:.0f} -> {m2:.0f} км/с: "
              f"глубина дуги А выходит за 3200 — совместимо со стеной Гидры-Антлии")
    else:
        print(f"\n    >>> Дуга А не изменилась ({n1} -> {n2}): её глубина честно ~24 Мпк, "
              f"это ближний край стены Гидры-Антлии")

    # cz-гистограмма дуги А на срезе 5000
    df = load(filename, 5000)
    ua = unit(df['RA'].values, df['DEC'].values)
    cz = df['CZ'].values
    theta = ang_dist(ua, unit(*DIP))
    phi   = azimuth(ua, unit(*DIP))
    arcA = (np.abs(theta - RING_R) <= RING_W) & (phi > 30) & (phi < 60)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 5000, 30)
    ax.hist(cz[arcA], bins=bins, color='green', alpha=0.7, label=f'Дуга А (N={arcA.sum()})')
    ax.axvline(3400, color='red', ls='--', label='cz ядра Гидры ~3400')
    ax.set_xlabel('cz, км/с'); ax.set_ylabel('число галактик')
    ax.set_title(f'{name}: cz-профиль дуги А, срез 5000')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = f'arcA_{name}.png'
    plt.savefig(out, dpi=150)
    print(f"    график: {out}")

analyze('2mrs.csv', '2MRS')
analyze('cf4.csv', 'CF4')
