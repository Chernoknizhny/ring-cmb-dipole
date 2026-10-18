import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

DIP   = (167.9, -7.2)
VIRGO = (187.7, 12.7)
RING_R, RING_W = 25.5, 2.0
D2R = np.pi / 180.0
S_RING = 2*np.pi*np.sin(np.radians(RING_R)) * 2*RING_W
S_BG   = 2*np.pi*np.sin(np.radians(17.5))*5 + 2*np.pi*np.sin(np.radians(32.5))*5

def unit(ra, dec):
    ra_, dec_ = ra * D2R, dec * D2R
    return np.stack([np.cos(dec_)*np.cos(ra_),
                     np.cos(dec_)*np.sin(ra_),
                     np.sin(dec_)], axis=-1)

def ang_dist(u1, u2):
    return np.arccos(np.clip(u1 @ u2, -1, 1)) / D2R

def basis(d):
    ref = np.array([0., 0., 1.]) if abs(d[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)

def azimuth(u, d):
    e1, e2 = basis(d)
    return np.degrees(np.arctan2(u @ e2, u @ e1)) % 360.0

def ratio(ring, bg, keep):
    n_r = (ring & keep).sum()
    n_b = (bg & keep).sum()
    return (n_r / S_RING) / (n_b / S_BG) if n_b > 0 else np.nan

def analyze(filename, name):
    df = pd.read_csv(filename)
    if 'DIST' in df.columns:
        df = df[df['DIST'] < 45]          # CF4: отбор по расстоянию
    else:
        df = df[df['CZ'] < 3200]          # 2MRS: отбор по cz
    ra, dec, cz = df['RA'].values, df['DEC'].values, df['CZ'].values
    print(f"\n{'='*55}\n{name}: N = {len(df)}\n{'='*55}")

    u_dip, u_vir = unit(*DIP), unit(*VIRGO)
    ua = unit(ra, dec)
    theta = ang_dist(ua, u_dip)
    phi   = azimuth(ua, u_dip)

    ring = np.abs(theta - RING_R) <= RING_W
    bg   = ((theta >= 15) & (theta <= 20)) | ((theta >= 30) & (theta <= 35))
    mask_v = ang_dist(ua, u_vir) < 10.0     # угловая маска Девы, без cz-ограничения

    R0 = ratio(ring, bg, np.ones(len(ua), dtype=bool))
    R1 = ratio(ring, bg, ~mask_v)
    print(f"[1] Ratio без маски: {R0:.2f}")
    print(f"    Ratio после маски Девы 10° (вся cz): {R1:.2f}  ({R1/R0*100:.0f}% сигнала)")
    print(f"    удалено из кольца: {(ring & mask_v).sum()} из {ring.sum()}")

    # --- моки: рандомизируем галактический азимут, маска считается в пространстве мока ---
    b = np.arcsin(np.clip(ua[:, 2], -1, 1)) / D2R
    rng = np.random.default_rng(42)
    mock0, mock1 = [], []
    for _ in range(1000):
        u_m = unit(rng.uniform(0, 360, len(ra)), b)
        th_m = ang_dist(u_m, u_dip)
        mv_m = ang_dist(u_m, u_vir) < 10.0
        r_m = np.abs(th_m - RING_R) <= RING_W
        g_m = ((th_m >= 15) & (th_m <= 20)) | ((th_m >= 30) & (th_m <= 35))
        mock0.append(ratio(r_m, g_m, np.ones(len(ra), dtype=bool)))
        mock1.append(ratio(r_m, g_m, ~mv_m))
    mock0, mock1 = np.array(mock0), np.array(mock1)
    print(f"    моки: без маски {mock0.mean():.2f}±{mock0.std():.2f} "
          f"| после маски {mock1.mean():.2f}±{mock1.std():.2f}")
    print(f"    p(без маски) = {max((mock0 > R0).mean(), 1/1000):.1e}   "
          f"p(после маски) = {max((mock1 > R1).mean(), 1/1000):.1e}")

    # --- дуга vs остаток ---
    arc  = ring & (phi > 210) & (phi < 240)
    rest = ring & ~arc
    print(f"\n[2] Дуга 210-240°: {arc.sum()} галактик, медиана cz = {np.median(cz[arc]):.0f} км/с")
    print(f"    Остаток кольца: {rest.sum()} галактик, медиана cz = {np.median(cz[rest]):.0f} км/с")
    if arc.sum() >= 5:
        print(f"    перц. 10-90 cz дуги: {np.percentile(cz[arc], [10, 90]).round(0)}")
    if arc.sum() > 10 and rest.sum() > 10:
        ks = ks_2samp(cz[arc], cz[rest])
        print(f"    KS дуга vs остаток: D = {ks.statistic:.2f}, p = {ks.pvalue:.1e}")
    if arc.sum() > 0:
        print(f"    доля дуги в конусе Девы 10°: {(mask_v[arc]).mean()*100:.0f}% "
              f"(остаток: {(mask_v[rest]).mean()*100:.0f}%)")

    # =========================================================================
    # [3] СЕКТОРНЫЙ АНАЛИЗ ОСТАТКА (вне конуса Девы) - ДОБАВЛЕННЫЙ БЛОК
    # =========================================================================
    print(f"\n[3] Секторный анализ остатка (вне Девы 10°):")
    res_mask = ring & ~mask_v
    n_res = res_mask.sum()
    
    if n_res > 0:
        bins_sec = np.arange(0, 361, 30)
        hist_sec, _ = np.histogram(phi[res_mask], bins=bins_sec)
        expect = n_res / 12
        
        for i, (a0, a1) in enumerate(zip(bins_sec[:-1], bins_sec[1:])):
            z = (hist_sec[i] - expect) / np.sqrt(max(expect, 1))
            bar = '#' * int(hist_sec[i])
            print(f"    {a0:3.0f}-{a1:3.0f}°: {hist_sec[i]:3d} {bar} (z={z:.1f})")
        print(f"    ожидание при равномерности: {expect:.1f} на сектор")

        i_max = int(np.argmax(hist_sec))
        c0, c1 = bins_sec[i_max], bins_sec[i_max+1]
        sel_sec = res_mask & (phi >= c0) & (phi < c1)
        
        if sel_sec.sum() > 5:
            # Перевод в небесные координаты
            u_c = ua[sel_sec].mean(0)
            u_c /= np.linalg.norm(u_c)
            dec_c = np.degrees(np.arcsin(np.clip(u_c[2], -1, 1)))
            ra_c = np.degrees(np.arctan2(u_c[1], u_c[0])) % 360
            med_cz = np.median(cz[sel_sec])
            
            print(f"    ПИК остатка: сектор {c0}-{c1}° ({sel_sec.sum()} гал.)")
            print(f"    RA = {ra_c:.1f}°, Dec = {dec_c:.1f}°")
            print(f"    медиана cz = {med_cz:.0f} км/с (~ {med_cz/70:.0f} Мпк)")

        # m-мода остатка
        phi_res = phi[res_mask]
        ms = np.arange(2, 13)
        z_m = np.array([np.abs(np.exp(1j * m * phi_res * D2R).sum())**2 / len(phi_res) for m in ms])
        rng_m = np.random.default_rng(555)
        P_mock_res = np.array([[np.abs(np.exp(1j * m * rng_m.uniform(0, 2*np.pi, n_res)).sum())**2 / n_res
                                for m in ms] for _ in range(2000)])
        Z_m = (z_m - P_mock_res.mean(0)) / np.where(P_mock_res.std(0) > 0, P_mock_res.std(0), 1)
        i_m = int(np.argmax(Z_m))
        p_m = max((P_mock_res[:, i_m] >= z_m[i_m]).mean(), 1/2000) * len(ms)
        
        print(f"    m-мода остатка: m = {ms[i_m]}, {Z_m[i_m]:.1f}σ, p = {p_m:.2f}")
    # =========================================================================

    # --- графики ---
    fig = plt.figure(figsize=(14, 5))
    ax0 = fig.add_subplot(1, 2, 1)
    bins = np.linspace(0, 3200, 25)
    ax0.hist(cz[arc],  bins=bins, alpha=0.55, color='red',  label=f'Дуга ({arc.sum()})', density=True)
    ax0.hist(cz[rest], bins=bins, alpha=0.55, color='blue', label=f'Остаток ({rest.sum()})', density=True)
    ax0.hist(cz[bg],   bins=bins, alpha=0.30, color='gray', label=f'Фон ({bg.sum()})', density=True)
    ax0.set_xlabel('cz, км/с'); ax0.set_ylabel('плотность')
    ax0.set_title(f'{name}: глубина дуги vs остатка')
    ax0.legend(); ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(1, 2, 2, projection='polar')
    ax1.scatter(np.radians(phi[ring]), cz[ring], s=14, alpha=0.6, c=cz[ring], cmap='viridis')
    ax1.set_title(f'{name}: азимут × cz в кольце')

    plt.tight_layout()
    out = f'final_test_{name}.png'
    plt.savefig(out, dpi=150)
    print(f"\nГрафик сохранен: {out}")

analyze('2mrs.csv', '2MRS')
analyze('cf4.csv', 'CF4')