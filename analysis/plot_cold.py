import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.style.use("seaborn-v0_8-whitegrid")
volumes = [16, 8, 12]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# COLD START

# Binder Cumulant U_4
plt.figure(figsize=(6, 4.5))
for L, color in zip(volumes, colors):
    try:
        df = pd.read_csv(f"su2_cold_L{L}.csv")
        ecolor = mcolors.to_rgba(color, alpha=0.4)
        plt.errorbar(
            df["beta"],
            df["U4"],
            yerr=df["U4_err"],
            fmt="-o",
            color=color,
            ecolor=ecolor,
            label=f"L = {L}",
            capsize=3,
            markersize=5,
        )
    except FileNotFoundError:
        print(f"Warning: su2_cold_L{L}.csv not found.")

plt.axvline(
    x=2.299,
    color="k",
    linestyle="--",
    alpha=0.5,
    label="Theory $\\beta_c \\approx 2.30$",
)
plt.title("Binder Cumulant Crossing", fontsize=12, fontweight="bold")
plt.xlabel("Coupling ($\\beta$)", fontsize=12)
plt.ylabel("Binder Cumulant $U_4$", fontsize=12)
plt.legend(loc="upper left", fontsize=12)
plt.tight_layout()

filename_u4 = "binder_cumulant_cold.png"
plt.savefig(filename_u4, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {filename_u4}")


# Finite-Size Scaling of Connected Susceptibility

plt.figure(figsize=(6, 4.5))
for L, color in zip(volumes, colors):
    try:
        df = pd.read_csv(f"su2_cold_L{L}.csv")
        ecolor = mcolors.to_rgba(color, alpha=0.4)
        plt.errorbar(
            df["beta"],
            df["chi"],
            yerr=df["chi_err"],
            fmt="-s",
            color=color,
            ecolor=ecolor,
            label=f"L = {L}",
            capsize=3,
            markersize=5,
        )
    except FileNotFoundError:
        pass
plt.axvline(
    x=2.299,
    color="k",
    linestyle="--",
    alpha=0.5,
    label="Theory $\\beta_c \\approx 2.30$",
)
plt.title("Connected Susceptibility Peak", fontsize=12, fontweight="bold")
plt.xlabel("Coupling ($\\beta$)", fontsize=12)
plt.ylabel("Susceptibility $\\chi_{conn}$", fontsize=12)
plt.legend(loc="upper left", fontsize=12)
plt.tight_layout()

filename_chi = "susceptibility_scaling_cold.png"
plt.savefig(filename_chi, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {filename_chi}")


# Polyakov Loop Expectation Value
plt.figure(figsize=(6, 4.5))
for L, color in zip(volumes, colors):
    try:
        df = pd.read_csv(f"su2_cold_L{L}.csv")
        ecolor = mcolors.to_rgba(color, alpha=0.4)
        plt.errorbar(
            df["beta"],
            df["L"],
            yerr=df["L_err"],
            fmt="-^",
            color=color,
            ecolor=ecolor,
            label=f"L = {L}",
            capsize=3,
            markersize=5,
        )
    except FileNotFoundError:
        pass

plt.axvline(
    x=2.299,
    color="k",
    linestyle="--",
    alpha=0.5,
    label="Theory $\\beta_c \\approx 2.30$",
)
plt.title("Polyakov Loop", fontsize=12, fontweight="bold")
plt.xlabel("Coupling ($\\beta$)", fontsize=12)
plt.ylabel("Polyakov Loop $\\langle P \\rangle$", fontsize=12)
plt.legend(loc="upper left", fontsize=12)
plt.tight_layout()

filename_polyakov = "polyakov_loop_cold.png"
plt.savefig(filename_polyakov, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {filename_polyakov}")

print("\nPlots generated successfully!")

# Credits to Gemini for the error bar plots

