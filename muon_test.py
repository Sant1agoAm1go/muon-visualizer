# Выполню загрузку и построю тепловые карты (сырое и скорректированное по эффективности).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython.display import display, Markdown  # For displaying DataFrames nicely

tracks_path = "download/npl4/1.0Grad/Tracks_DistrOutput_1.dat"
eff_path = "download/npl4/1.0Grad/EffCorFile_Tracks.dat"

# Проверим существование файлов
for p in (tracks_path, eff_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Файл не найден: {p}")

# Загрузим tracks: три колонки (Tetta, Phi, Count) - пробел/таб
tracks = pd.read_csv(tracks_path, sep=r"\s+", header=None, names=["Tetta","Phi","Count"], engine="python")
# Убедимся в типах
tracks["Tetta"] = tracks["Tetta"].astype(float)
tracks["Phi"] = tracks["Phi"].astype(float)
tracks["Count"] = tracks["Count"].astype(float)

# Загрузим eff: колонки: degree, rad, calib_counts, correction (multiplier)
eff = pd.read_csv(eff_path, sep=r"\s+", header=None, names=["T_deg","T_rad","calib_counts","corr"], engine="python")
eff["T_deg"] = eff["T_deg"].astype(float)
eff["corr"] = eff["corr"].astype(float)

# Интерполяция коррекции по углу Tetta (в градусах).
# Tracks.Tetta соответствует первой колонке eff.T_deg (в градусах) — но возможны несоответствия, используем интерполяцию.
t_vals = eff["T_deg"].values
c_vals = eff["corr"].values

# Ограничим область интерполяции: вне диапазона используем крайние значения
def get_corr(theta_array):
    # numpy.interp требует возрастающий t_vals
    return np.interp(theta_array, t_vals, c_vals, left=c_vals[0], right=c_vals[-1])

tracks["corr"] = get_corr(tracks["Tetta"])
tracks["Count_corr"] = tracks["Count"] * tracks["corr"]

# Соберём в 2D матрицы (Tetta x Phi). Определим диапазоны
tetta_vals = np.sort(tracks["Tetta"].unique())
phi_vals = np.sort(tracks["Phi"].unique())

Tn = len(tetta_vals); Pn = len(phi_vals)
# Создаём матрицы заполненные NaN (для отсутствующих комбинаций)
mat_raw = np.full((Tn, Pn), np.nan)
mat_corr = np.full((Tn, Pn), np.nan)

tetta_to_idx = {t:i for i,t in enumerate(tetta_vals)}
phi_to_idx = {p:i for i,p in enumerate(phi_vals)}

for _, row in tracks.iterrows():
    i = tetta_to_idx[row["Tetta"]]
    j = phi_to_idx[row["Phi"]]
    mat_raw[i,j] = row["Count"]
    mat_corr[i,j] = row["Count_corr"]

# Заменим NaN на 0 (бин с 0 треков)
mat_raw = np.nan_to_num(mat_raw, nan=0.0)
mat_corr = np.nan_to_num(mat_corr, nan=0.0)

# Построим две тепловые карты: сырой и скорректированный
fig, axs = plt.subplots(1,2, figsize=(14,6))
im0 = axs[0].imshow(mat_raw, origin='lower', aspect='auto', extent=[phi_vals[0], phi_vals[-1], tetta_vals[0], tetta_vals[-1]])
axs[0].set_title("Сырой биннинг: Count")
axs[0].set_xlabel("Phi (deg)")
axs[0].set_ylabel("Tetta (deg)")
fig.colorbar(im0, ax=axs[0], label="Count")

im1 = axs[1].imshow(mat_corr, origin='lower', aspect='auto', extent=[phi_vals[0], phi_vals[-1], tetta_vals[0], tetta_vals[-1]])
axs[1].set_title("Скорректировано по EffCorFile: Count * corr")
axs[1].set_xlabel("Phi (deg)")
axs[1].set_ylabel("Tetta (deg)")
fig.colorbar(im1, ax=axs[1], label="Corrected Count")

plt.tight_layout()
plt.show()

# Выведем статистику и топ-10 бинов по скорректированному счёту
total_raw = mat_raw.sum()
total_corr = mat_corr.sum()
stats = {
    "total_raw": total_raw,
    "total_corrected": total_corr,
    "Tetta_bins": Tn,
    "Phi_bins": Pn
}

stats_df = pd.DataFrame([stats])

# Топ-10
tracks_top10 = tracks.sort_values("Count_corr", ascending=False).head(10).reset_index(drop=True)

# ===== REPLACED THE PROPRIETARY DISPLAY FUNCTIONS =====
# Display statistics
print("\nСтатистика распределения:")
display(stats_df)

print("\nТоп-10 бинов (скорректированные):")
display(tracks_top10)

# Также сохраним матрицу скорректированного распределения в CSV для дальнейшего использования
out_csv = "Tracks_DistrOutput_1_corrected.csv"
df_out = tracks[["Tetta","Phi","Count","corr","Count_corr"]]
df_out.to_csv(out_csv, index=False)

print(f"\nСкорректированные данные сохранены в: {out_csv}")

