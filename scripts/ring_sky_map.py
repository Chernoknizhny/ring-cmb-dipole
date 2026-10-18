import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIP   = (167.9, -7.2)
VIRGO = (187.7, 12.7)
RING_R, RING_W = 25.5, 2.0
D2R = np.pi / 180.0

# Известные структуры: имя -> (RA, Dec, cz, км/с)
CLUSTERS = {
    "Дева (M87)":        (187.7,  12.7, 1300),
    "Гидра (A1060)":     (159.3, -27.5, 3400),
    "Антлия":            (157.5, -39.0, 3000),
    "Центавр (A3526)":   (192.3, -41.5, 3040),
    "Форнакс":           ( 54.6, -35.4, 1440),
    "Эридан":            ( 57.0, -21.0, 1900),
    "М81-группа":        (148.9,  69.0,  -30),
}

# Галактические координаты: экватор -> гал (IAU)
M_EQ2GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                     [ 0.4941094279, -0.4448296300,  0.7469822445],
                     [-0.8676661490, -0.1980763734,  0.4559837762]])

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

def rotate_around(u, d, alpha):
    """Поворот векторов u на угол alpha (рад) вокруг оси d."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    return u*ca + np.cross(np.broadcast_to(d, u.shape), u)*sa + d*(u @ d)[:, None]*(1 - ca)

def gal_b(u):
    g = u @ M_EQ2GAL.T
    return np.degrees(np.arcsin(np.clip(g[:, 2], -1, 1)))

def load(filename):
    df = pd.read_csv(filename)
    if 'DIST' in df.columns:
        df = df[(df['DIST'] > 0) & (df['DIST'] < 45)]
        dist = df['DIST'].values
    else:
        df = df[(df['CZ'] > 0) & (df['CZ'] < 3200)]
        dist = None
    return df, dist

def analyze(filename, name):
    df, dist = load(filename)
    ra, dec, cz = df['RA'].values, df['DEC'].values, df['CZ'].values
    print(f"\n{'='*60}\n{name}: N = {len(df)} (глубина: "
          f"{'DIST, Мпк' if dist is not None else 'cz'})\n{'='*60}")

    u_dip = unit(*DIP)
    ua = unit(ra, dec)
    theta = ang_dist(ua, u_dip)
    phi   = azimuth(ua, u_dip)
    ring  = np.abs(theta - RING_R) <= RING_W
    arc   = ring & (phi > 210) & (phi < 240)
    mask_v  = ang_dist(ua, unit(*VIRGO)) < 10.0
    res   = ring & ~mask_v          # остаток (вне конуса Девы)
    arcA  = ring & (phi > 30)  & (phi < 60)   # дуга А
    arcB  = ring & (phi > 180) & (phi < 210)  # дуга Б

    # ---------- [1] Кросс-матч дуг со скоплениями ----------
    print("\n[1] Доля галактик каждой структуры в конусе 10° вокруг скоплений:")
    hdr = "    структура".ljust(22) + "".join(f"{c:>14s}" for c in ("дуга 210-240", "дуга А 30-60", "дуга Б 180-210", "остаток"))
    print(hdr)
    for cname, (cra, cdec, ccz) in CLUSTERS.items():
        mc = ang_dist(ua, unit(cra, cdec)) < 10.0
        print(f"    {cname:<20s}"
              + "".join(f"{(mc[s].mean()*100 if s.sum() else np.nan):>13.0f}%" for s in (arc, arcA, arcB, res)))
    print("    (строка = скопление; процент = доля галактик структуры в 10° от него)")

    # ---------- [2] Глубины: CF4-расстояния либо cz ----------
    print("\n[2] Глубина структур (медиана):")
    for lab, sel in (("дуга 210-240 (Дева)", arc), ("дуга А 30-60", arcA),
                     ("дуга Б 180-210", arcB), ("остаток", res)):
        if sel.sum() == 0: continue
        if dist is not None:
            med = np.median(dist[sel]); unit_ = "Мпк (CF4)"
        else:
            med = np.median(cz[sel]); unit_ = "км/с"
        print(f"    {lab:<22s}: {med:6.1f} {unit_}  (N={sel.sum()})")
    print("    эталоны: Дева ~16-20 Мпк, Гидра ~45 Мпк (cz~3400), Антлия ~40 Мпк (cz~3000), "
          "Форнакс ~19 Мпк (cz~1440)")

    # ---------- [3] Честный нуль для m-моды: поворот вокруг диполя ----------
    print("\n[3] m-мода остатка, нуль = поворот каталога вокруг оси диполя:")
    phi_res = azimuth(ua[res], u_dip)
    ms = np.arange(2, 13)
    def pmod(ph, mmax=ms):
        return np.array([np.abs(np.exp(1j*m*ph*D2R).sum())**2/len(ph) for m in mmax])
    P_obs = pmod(phi_res)
    rng = np.random.default_rng(31337)
    P_rot = np.empty((500, len(ms)))
    for i in range(500):
        a = rng.uniform(0, 2*np.pi)
        u_rot = rotate_around(ua, u_dip, a)
        keep = np.abs(gal_b(u_rot)) > 5.0          # зона избегания в новом положении
        th_r = ang_dist(u_rot, u_dip)
        r_r = (np.abs(th_r - RING_R) <= RING_W) & keep & ~((ang_dist(u_rot, unit(*VIRGO)) < 10.0))
        if r_r.sum() < 10:
            P_rot[i] = np.nan; continue
        P_rot[i] = pmod(azimuth(u_rot[r_r], u_dip))
    P_rot = P_rot[~np.isnan(P_rot).any(1)]
    Z = (P_obs - P_rot.mean(0)) / np.where(P_rot.std(0) > 0, P_rot.std(0), 1)
    i_m = int(np.argmax(Z))
    p_m = min(1.0, max((P_rot[:, i_m] >= P_obs[i_m]).mean(), 1/len(P_rot)) * len(ms))
    print(f"    m = {ms[i_m]}, {Z[i_m]:.1f}σ, p(с поправкой на {len(ms)} гармоник) = {p_m:.3f}")
    print("    (нулём служит реальная селекция: поворот сохраняет зону избегания и маски)")

    # ---------- [4] Карта неба ----------
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 1, 1)
    b_all = gal_b(ua)
    ok = np.abs(b_all) > 5
    ax.scatter(ra[ok & ~ring], dec[ok & ~ring], s=4, c='lightgray', label='фон')
    ax.scatter(ra[ring & ~arc], dec[ring & ~arc], s=22, c='blue', label='кольцо вне дуги Девы')
    ax.scatter(ra[arcA], dec[arcA], s=34, c='green', label='дуга А 30-60°')
    ax.scatter(ra[arcB], dec[arcB], s=34, c='purple', label='дуга Б 180-210°')
    ax.scatter(ra[arc], dec[arc], s=34, c='red', label='дуга Девы 210-240°')
    for cname, (cra, cdec, ccz) in CLUSTERS.items():
        ax.scatter(cra, cdec, marker='*', s=280, c='gold', edgecolors='k', zorder=5)
        ax.annotate(cname, (cra, cdec), fontsize=9, xytext=(5, 5), textcoords='offset points')
    ax.scatter(*DIP, marker='x', s=120, c='black', label='диполь СМВ')
    ax.set_xlabel('RA, град'); ax.set_ylabel('Dec, град')
    ax.set_xlim(360, 0); ax.set_ylim(-60, 75)     # RA убывает вправо, как на картах неба
    ax.set_title(f'{name}: дуги кольца и известные скопления')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    out = f'sky_map_{name}.png'
    plt.savefig(out, dpi=150)
    print(f"\nКарта сохранена: {out}")

analyze('2mrs.csv', '2MRS')
analyze('cf4.csv', 'CF4')
