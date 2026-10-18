import pandas as pd
from astropy.io import fits

def convert(fits_file, csv_file):
    print(f"Читаю {fits_file}...")
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        
        # Извлекаем RA и DEC
        ra = data['RAJ2000']
        dec = data['DEJ2000']
        
        # Ищем скорость (CZ) или расстояние (DM)
        if 'cz' in data.names:
            cz = data['cz']
        elif 'Vcmb' in data.names:  # на случай если в CF4 колонка называется Vcmb
            cz = data['Vcmb']
        elif 'DM' in data.names:
            # Если есть только расстояние, переводим его в cz (грубо) для скрипта
            dm = data['DM']
            d_mpc = 10**((dm + 5)/5) / 1e6
            cz = d_mpc * 70.0 # H0 = 70 км/с/Мпк
        else:
            print(f"ОШИБКА: не найдена колонка со скоростью или расстоянием в {fits_file}")
            return

        df = pd.DataFrame({'RA': ra, 'DEC': dec, 'CZ': cz})
        # Убираем пустые значения
        df = df.dropna()
        df.to_csv(csv_file, index=False)
        print(f"Сохранено: {csv_file} (строк: {len(df)})")

convert('2mrs_full.fits', '2mrs.csv')
convert('cf4.fits', 'cf4.csv')