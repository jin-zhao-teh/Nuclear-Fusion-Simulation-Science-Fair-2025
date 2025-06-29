import numpy as np
np.seterr(divide='raise', invalid='raise', over='raise', under='ignore')

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- CONSTANTS ---
keV_to_J = 1.602e-16
e = 1.602e-19
mu0 = 4 * np.pi * 1e-7

# --- DEVICE PARAMETERS (all devices included) ---
devices = {
    "Tokamak": {
        "n0": 5e19, "T0": 10, "V": 100, "fuel": "DT", "I_p": 1e6, "B0": 5, "R": 2, "a": 0.5,
        "kappa": 1.7, "H_98": 1.0, "impurity": "C", "Z_imp": 6, "Z_eff": 1.5, "impurity_frac": 0.01,
    },
    "ITER": {
        "n0": 1e20, "T0": 20, "V": 840, "fuel": "DT", "I_p": 1.5e6, "B0": 5.3, "R": 6.2, "a": 2,
        "kappa": 1.7, "H_98": 1.0, "impurity": "W", "Z_imp": 74, "Z_eff": 1.5, "impurity_frac": 0.01,
    },
    "Polaris": {
        "n0": 5e22, "T0": 20, "V": 0.5, "fuel": "DHe3", "I_p": 0, "B0": 10, "R": 0.5, "a": 0.1,
        "kappa": 1.0, "H_98": 1.0, "impurity": "C", "Z_imp": 6, "Z_eff": 2.0, "impurity_frac": 0.02,
    },
    "LDX Junior": {
        "n0": 1e19, "T0": 1, "V": 0.014, "fuel": "DD", "I_p": 0, "B0": 0.5, "R": 0.2, "a": 0.05,
        "kappa": 1.0, "H_98": 1.0, "impurity": "C", "Z_imp": 6, "Z_eff": 1.2, "impurity_frac": 0.005,
    },
    "Stellarator": {
        "n0": 2e19, "T0": 8, "V": 30, "fuel": "DT", "I_p": 0, "B0": 3, "R": 1.5, "a": 0.4,
        "kappa": 1.7, "H_98": 1.0, "impurity": "C", "Z_imp": 6, "Z_eff": 1.3, "impurity_frac": 0.008,
    },
    "Z-pinch": {
        "n0": 1e21, "T0": 5, "V": 0.01, "fuel": "DT", "I_p": 2e6, "B0": 20, "R": 0.01, "a": 0.005,
        "kappa": 1.0, "H_98": 1.0, "impurity": "C", "Z_imp": 6, "Z_eff": 1.7, "impurity_frac": 0.03,
    },
    "ICF": {
        "n0": 1e26, "T0": 10, "V": 1e-9, "fuel": "DT", "I_p": 0, "B0": 0, "R": 0.001, "a": 0.0005,
        "kappa": 1.0, "H_98": 1.0, "impurity": "C", "Z_imp": 6, "Z_eff": 1.1, "impurity_frac": 0.001,
    }
}

def sigma_v(T, fuel):
    T = np.maximum(T, 0.1)
    if fuel == "DT":
        exponent = -((T / 69)**0.8)
        exponent = np.clip(exponent, -700, 700)
        return 3.68e-18 * (T / 17.6)**0.3 * np.exp(exponent)
    elif fuel == "DHe3":
        exponent = -18.8 * T**(-1/3)
        exponent = np.clip(exponent, -700, 700)
        return 2.5e-21 * T**(-2/3) * np.exp(exponent)
    elif fuel == "DD":
        exponent = -18.76 * T**(-1/3)
        exponent = np.clip(exponent, -700, 700)
        return 1.6e-22 * T**(-2/3) * np.exp(exponent)
    else:
        return 0.0

def fusion_energy(fuel):
    if fuel == "DT": return 17.6e6 * e
    elif fuel == "DHe3": return 18.3e6 * e
    elif fuel == "DD": return 3.65e6 * e
    else: return 0.0

def P_brem(n, T, Z_eff):
    return 1.69e-38 * Z_eff * n**2 * np.sqrt(T)

def P_line(n, T, Z_imp, impurity_frac):
    return 1e-37 * Z_imp**2 * impurity_frac * n**2 * np.exp(-T/10)

def D_gyrobohm(T, B):
    return 0.1 * (T**1.5) / (B + 1e-6)

# --- Live 1D Transport Model ---
class LiveSim:
    def __init__(self, params, n_r=30, dt=0.1):
        self.params = params
        self.a = params["a"]
        self.R = params["R"]
        self.B = params["B0"]
        self.fuel = params["fuel"]
        self.Z_eff = params["Z_eff"]
        self.Z_imp = params["Z_imp"]
        self.impurity_frac = params["impurity_frac"]
        self.n_r = n_r
        self.dt = dt
        self.r = np.linspace(0, self.a, n_r)
        self.dr = self.r[1] - self.r[0]
        self.T = np.ones(n_r) * params["T0"] * (1 - (self.r/self.a)**2 * 0.8)
        self.n = np.ones(n_r) * params["n0"]
        self.time = [0.0]
        self.T_central = [self.T[0]]
        self.T_edge = [self.T[-1]]
        self.n_central = [self.n[0]]
        self.n_edge = [self.n[-1]]
        self.P_fusion_total = [0.0]

    def step(self):
        D = D_gyrobohm(self.T, self.B)
        chi = np.clip(D.copy(), 0, 1e3)  # Cap chi to prevent runaway
        sv = sigma_v(self.T, self.fuel)
        E_fus = fusion_energy(self.fuel)
        if self.fuel == "DT":
            n_D = n_T = self.n / 2
            P_fusion = n_D * n_T * sv * E_fus
            f_alpha = 0.2
        else:
            P_fusion = 0.25 * self.n**2 * sv * E_fus
            f_alpha = 0.0
        dV = 2 * np.pi * self.R * self.r * self.dr
        self.P_fusion_total.append(np.sum(P_fusion * dV))
        P_alpha = f_alpha * P_fusion
        P_brem_arr = P_brem(self.n, self.T, self.Z_eff)
        P_line_arr = P_line(self.n, self.T, self.Z_imp, self.impurity_frac)
        P_rad = P_brem_arr + P_line_arr

        n_new = self.n.copy()
        for i in range(1, self.n_r-1):
            n_new[i] = self.n[i] + self.dt * (
                (D[i+1]*(self.n[i+1]-self.n[i])/self.dr - D[i]*(self.n[i]-self.n[i-1])/self.dr) / self.dr
            )
        n_new[0] = n_new[1]
        n_new[-1] = n_new[-2]

        T_new = self.T.copy()
        for i in range(1, self.n_r-1):
            T_new[i] = self.T[i] + self.dt * (
                (chi[i+1]*(self.T[i+1]-self.T[i])/self.dr - chi[i]*(self.T[i]-self.T[i-1])/self.dr) / self.dr
                + (P_alpha[i] - P_rad[i]) * self.dt / (1.5 * self.n[i] * keV_to_J)
            )
        T_new[0] = T_new[1]
        T_new[-1] = T_new[-2]

        # Cap temperature to prevent runaway
        T_new = np.clip(T_new, 1e-3, 1e3)
        n_new = np.clip(n_new, 1e-10, 1e30)

        # Check for numerical instability
        if not np.all(np.isfinite(T_new)) or not np.all(np.isfinite(n_new)):
            print("Numerical instability detected: T_new or n_new contains NaN or Inf. Skipping update.")
            return

        self.n = n_new
        self.T = T_new
        self.time.append(self.time[-1] + self.dt)
        self.T_central.append(self.T[0])
        self.T_edge.append(self.T[-1])
        self.n_central.append(self.n[0])
        self.n_edge.append(self.n[-1])

    def get_results(self):
        return {
            "r": self.r, "T": self.T, "n": self.n,
            "T_central": np.array(self.T_central), "T_edge": np.array(self.T_edge),
            "n_central": np.array(self.n_central), "n_edge": np.array(self.n_edge),
            "P_fusion_total": np.array(self.P_fusion_total),
            "time": np.array(self.time)
        }

# --- Tkinter UI and Live Update ---
root = tk.Tk()
root.title("Fusion Reactor 1D Simulation (Live)")
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# Initialize live simulators for all devices
sims = {dev: LiveSim(params) for dev, params in devices.items()}

# --- Home Tab: Fusion Power Output with Sticky Note ---
home_tab = ttk.Frame(notebook)
notebook.add(home_tab, text="Home: Fusion Power Output")

# Sticky note label
sticky_label = tk.Label(home_tab, text="", font=("Segoe UI", 12), bg="#ffffcc", anchor="w", justify="left", relief="solid", bd=2)
sticky_label.pack(fill=tk.X, padx=10, pady=5)

fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=100)
lines_Pfusion = {}
for dev, sim in sims.items():
    res = sim.get_results()
    (lines_Pfusion[dev],) = ax3.plot(res["time"], res["P_fusion_total"], label=dev)
ax3.set_title("Total Fusion Power Output vs Time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Fusion Power (W)")
ax3.legend()
fig3.tight_layout()
canvas3 = FigureCanvasTkAgg(fig3, master=home_tab)
canvas3.draw()
canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=1)

# --- Central/Edge Evolution Tab ---
evol_tab = ttk.Frame(notebook)
notebook.add(evol_tab, text="Central/Edge Evolution")
fig, axs = plt.subplots(2, 1, figsize=(8, 7), dpi=100)
lines_Tc, lines_Te, lines_nc, lines_ne = {}, {}, {}, {}
for dev, sim in sims.items():
    res = sim.get_results()
    (lines_Tc[dev],) = axs[0].plot(res["time"], res["T_central"], label=f"{dev} (center)")
    (lines_Te[dev],) = axs[0].plot(res["time"], res["T_edge"], '--', label=f"{dev} (edge)")
    (lines_nc[dev],) = axs[1].plot(res["time"], res["n_central"], label=f"{dev} (center)")
    (lines_ne[dev],) = axs[1].plot(res["time"], res["n_edge"], '--', label=f"{dev} (edge)")
axs[0].set_title("Central/Edge Temperature Evolution")
axs[0].set_ylabel("T (keV)")
axs[0].legend()
axs[1].set_title("Central/Edge Density Evolution")
axs[1].set_ylabel("n (m^-3)")
axs[1].set_xlabel("Time (s)")
axs[1].legend()
fig.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=evol_tab)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

# --- Final Profiles Tab ---
prof_tab = ttk.Frame(notebook)
notebook.add(prof_tab, text="Final Profiles")
fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
lines_Tprof, lines_nprof = {}, {}
for dev, sim in sims.items():
    res = sim.get_results()
    (lines_Tprof[dev],) = axs2[0].plot(res["r"], res["T"], label=dev)
    (lines_nprof[dev],) = axs2[1].plot(res["r"], res["n"], label=dev)
axs2[0].set_title("Final Temperature Profile")
axs2[0].set_xlabel("r (m)")
axs2[0].set_ylabel("T (keV)")
axs2[0].legend()
axs2[1].set_title("Final Density Profile")
axs2[1].set_xlabel("r (m)")
axs2[1].set_ylabel("n (m^-3)")
axs2[1].legend()
fig2.tight_layout()
canvas2 = FigureCanvasTkAgg(fig2, master=prof_tab)
canvas2.draw()
canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=1)

# --- Live Update Loop ---
def live_update():
    sticky_text = "Average Fusion Power Output (W):\n"
    for dev, sim in sims.items():
        for _ in range(5):  # Advance by 5 steps per GUI update for speed
            sim.step()
        res = sim.get_results()
        # Update fusion power plot (Home tab)
        lines_Pfusion[dev].set_data(res["time"], res["P_fusion_total"])
        ax3.set_xlim(0, res["time"][-1])
        # Update sticky note
        avg_power = np.mean(res["P_fusion_total"])
        sticky_text += f"{dev}: {avg_power:.3e} W\n"
        # Update evolution plots
        lines_Tc[dev].set_data(res["time"], res["T_central"])
        lines_Te[dev].set_data(res["time"], res["T_edge"])
        lines_nc[dev].set_data(res["time"], res["n_central"])
        lines_ne[dev].set_data(res["time"], res["n_edge"])
        axs[0].set_xlim(0, res["time"][-1])
        axs[1].set_xlim(0, res["time"][-1])
        # Update profiles
        lines_Tprof[dev].set_data(res["r"], res["T"])
        lines_nprof[dev].set_data(res["r"], res["n"])
        # Print debug info
        print(f"{dev}: t={res['time'][-1]:.2f}s, T0={res['T_central'][-1]:.2f}keV, n0={res['n_central'][-1]:.2e}, Pfus={res['P_fusion_total'][-1]:.2e}W")
    sticky_label.config(text=sticky_text.strip())
    axs[0].relim(); axs[0].autoscale_view()
    axs[1].relim(); axs[1].autoscale_view()
    axs2[0].relim(); axs2[0].autoscale_view()
    axs2[1].relim(); axs2[1].autoscale_view()
    ax3.relim(); ax3.autoscale_view()
    canvas.draw()
    canvas2.draw()
    canvas3.draw()
    root.after(100, live_update)  # update every 100 ms

live_update()
root.mainloop()