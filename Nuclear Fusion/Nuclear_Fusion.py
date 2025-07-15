import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm

# ================================================
# STEP 1: PHYSICS CONSTANTS AND FORMULAS
# ================================================
keV_to_J = 1.602e-16  # keV to Joules conversion
e = 1.602e-19         # Elementary charge (C)
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)

# Fusion reaction cross-sections
def sigma_v(T, fuel):
    T = np.maximum(T, 0.1)  # Minimum temperature = 0.1 keV
    if fuel == "DT":
        exponent = -((T / 69)**0.8)
        exponent = np.clip(exponent, -700, 700)  # Prevent overflow
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

# Fusion energy per reaction
def fusion_energy(fuel):
    if fuel == "DT": return 17.6e6 * e  # Joules
    elif fuel == "DHe3": return 18.3e6 * e
    elif fuel == "DD": return 3.65e6 * e
    else: return 0.0

# Radiation losses
def P_brem(n, T, Z_eff):
    """Bremsstrahlung radiation (W/m³)"""
    return 1.69e-38 * Z_eff * n**2 * np.sqrt(T)

def P_line(n, T, Z_imp, impurity_frac):
    """Line radiation (W/m³)"""
    return 1e-37 * Z_imp**2 * impurity_frac * n**2 * np.exp(-T/10)

# Transport model
def D_gyrobohm(T, B):
    """Gyro-Bohm diffusion coefficient (m²/s)"""
    return 0.1 * (T**1.5) / (B + 1e-6)  # Avoid division by zero

# ================================================
# STEP 2: DEVICE PARAMETERS
# ================================================
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

# ================================================
# STEP 3: SIMULATION CLASSES
# ================================================
class LiveSim:
    """1D Radial Transport Simulation"""
    def __init__(self, params, n_r=30, dt=0.1):
        self.params = params
        self.a = params["a"]  # Minor radius
        self.R = params["R"]  # Major radius
        self.B = params["B0"]
        self.fuel = params["fuel"]
        self.Z_eff = params["Z_eff"]
        self.Z_imp = params["Z_imp"]
        self.impurity_frac = params["impurity_frac"]
        self.n_r = n_r
        self.dt = dt
        
        # Radial grid
        self.r = np.linspace(0, self.a, n_r)
        self.dr = self.r[1] - self.r[0]
        
        # Initialize profiles with parabolic shapes
        self.T = np.ones(n_r) * params["T0"] * (1 - (self.r/self.a)**2 * 0.8)
        self.n = np.ones(n_r) * params["n0"]
        
        # Time evolution tracking
        self.time = [0.0]
        self.T_central = [self.T[0]]
        self.T_edge = [self.T[-1]]
        self.n_central = [self.n[0]]
        self.n_edge = [self.n[-1]]
        self.P_fusion_total = [0.0]
        self.E_fusion_total = [0.0]  # Cumulative energy

    def step(self):
        """Advance simulation by one time step"""
        # Calculate transport coefficients
        D = D_gyrobohm(self.T, self.B)
        chi = np.clip(D, 0, 1e3)  # Prevent excessive values
        
        # Calculate fusion power
        sv = sigma_v(self.T, self.fuel)
        E_fus = fusion_energy(self.fuel)
        
        if self.fuel == "DT":
            n_D = n_T = self.n / 2
            P_fusion = n_D * n_T * sv * E_fus
            f_alpha = 0.2  # Alpha particle fraction
        else:
            P_fusion = 0.25 * self.n**2 * sv * E_fus
            f_alpha = 0.0
        
        # Total power calculation
        dV = 2 * np.pi * self.R * self.r * self.dr  # Volume element
        total_power = np.sum(P_fusion * dV)
        self.P_fusion_total.append(total_power)
        self.E_fusion_total.append(self.E_fusion_total[-1] + total_power * self.dt)
        
        # Heating sources
        P_alpha = f_alpha * P_fusion
        P_aux = 1e7  # 10 MW auxiliary heating
        
        # Radiation losses
        P_brem_arr = P_brem(self.n, self.T, self.Z_eff)
        P_line_arr = P_line(self.n, self.T, self.Z_imp, self.impurity_frac)
        P_rad = P_brem_arr + P_line_arr
        
        # Fueling term (keeps density from dropping to zero)
        fueling = 0.01 * self.params["n0"]
        
        # Update density profile (diffusion + fueling)
        n_new = self.n.copy()
        for i in range(1, self.n_r-1):
            flux_in = D[i+1] * (self.n[i+1] - self.n[i]) / self.dr
            flux_out = D[i] * (self.n[i] - self.n[i-1]) / self.dr
            n_new[i] = self.n[i] + self.dt * ((flux_in - flux_out) / self.dr + fueling)
        
        # Boundary conditions (zero gradient)
        n_new[0] = n_new[1]
        n_new[-1] = n_new[-2]
        
        # Update temperature profile (conduction + heating - radiation)
        T_new = self.T.copy()
        for i in range(1, self.n_r-1):
            heat_in = chi[i+1] * (self.T[i+1] - self.T[i]) / self.dr
            heat_out = chi[i] * (self.T[i] - self.T[i-1]) / self.dr
            heat_source = (P_alpha[i] - P_rad[i] + P_aux) / (1.5 * self.n[i] * keV_to_J)
            T_new[i] = self.T[i] + self.dt * ((heat_in - heat_out) / self.dr + heat_source)
        
        # Boundary conditions
        T_new[0] = T_new[1]
        T_new[-1] = T_new[-2]
        
        # Numerical stability
        T_new = np.clip(T_new, 1e-3, 1e3)
        n_new = np.clip(n_new, 1e-10, 1e30)
        
        if not np.all(np.isfinite(T_new)) or not np.all(np.isfinite(n_new)):
            print("Numerical instability detected. Skipping update.")
            return
        
        # Update state
        self.n = n_new
        self.T = T_new
        self.time.append(self.time[-1] + self.dt)
        self.T_central.append(self.T[0])
        self.T_edge.append(self.T[-1])
        self.n_central.append(self.n[0])
        self.n_edge.append(self.n[-1])

    def get_results(self):
        """Return current simulation results"""
        return {
            "r": self.r, "T": self.T, "n": self.n,
            "T_central": np.array(self.T_central), "T_edge": np.array(self.T_edge),
            "n_central": np.array(self.n_central), "n_edge": np.array(self.n_edge),
            "P_fusion_total": np.array(self.P_fusion_total),
            "E_fusion_total": np.array(self.E_fusion_total),
            "time": np.array(self.time)
        }

class LiveSim2D:
    """2D R-Z Transport Simulation"""
    def __init__(self, params, n_r=32, n_z=32, dt=0.05):
        self.params = params
        self.a = params["a"]
        self.R0 = params["R"]  # Major radius
        self.B = params["B0"]
        self.fuel = params["fuel"]
        self.Z_eff = params["Z_eff"]
        self.Z_imp = params["Z_imp"]
        self.impurity_frac = params["impurity_frac"]
        self.n_r = n_r
        self.n_z = n_z
        self.dt = dt
        
        # Create R-Z grid
        self.r = np.linspace(0, self.a, n_r)
        self.z = np.linspace(-self.a, self.a, n_z)
        R, Z = np.meshgrid(self.r, self.z, indexing='ij')
        
        # Initialize profiles
        self.T = np.ones((n_r, n_z)) * params["T0"] * (1 - (R/self.a)**2 * 0.8)
        self.n = np.ones((n_r, n_z)) * params["n0"]
        
        # Time tracking
        self.time = [0.0]
        self.T_central = [self.T[n_r//2, n_z//2]]
        self.n_central = [self.n[n_r//2, n_z//2]]
        self.P_fusion_total = [0.0]
        self.E_fusion_total = [0.0]

    def step(self):
        """Advance simulation by one time step"""
        # Transport coefficients
        D = D_gyrobohm(self.T, self.B)
        chi = np.clip(D, 0, 1e3)
        
        # Fusion power
        sv = sigma_v(self.T, self.fuel)
        E_fus = fusion_energy(self.fuel)
        
        if self.fuel == "DT":
            n_D = n_T = self.n / 2
            P_fusion = n_D * n_T * sv * E_fus
            f_alpha = 0.2
        else:
            P_fusion = 0.25 * self.n**2 * sv * E_fus
            f_alpha = 0.0
        
        # Total power calculation
        dA = 2 * np.pi * self.R0 * (self.a/self.n_r) * (2*self.a/self.n_z)  # Area element
        total_power = np.sum(P_fusion * dA)
        self.P_fusion_total.append(total_power)
        self.E_fusion_total.append(self.E_fusion_total[-1] + total_power * self.dt)
        
        # Heating and radiation
        P_alpha = f_alpha * P_fusion
        P_aux = 1e7  # 10 MW
        P_brem_arr = P_brem(self.n, self.T, self.Z_eff)
        P_line_arr = P_line(self.n, self.T, self.Z_imp, self.impurity_frac)
        P_rad = P_brem_arr + P_line_arr
        
        # Fueling
        fueling = 0.01 * self.params["n0"]
        
        # Grid spacing
        dr = self.a / (self.n_r - 1)
        dz = 2 * self.a / (self.n_z - 1)
        
        # Update profiles
        n_new = self.n.copy()
        T_new = self.T.copy()
        
        for i in range(1, self.n_r-1):
            for j in range(1, self.n_z-1):
                # Density update (diffusion + fueling)
                n_flux_r = (D[i+1, j]*(self.n[i+1, j]-self.n[i, j])/dr - D[i, j]*(self.n[i, j]-self.n[i-1, j])/dr)/dr
                n_flux_z = (D[i, j+1]*(self.n[i, j+1]-self.n[i, j])/dz - D[i, j]*(self.n[i, j]-self.n[i, j-1])/dz)/dz
                n_new[i, j] = self.n[i, j] + self.dt * (n_flux_r + n_flux_z + fueling)
                
                # Temperature update (conduction + heating - radiation)
                T_flux_r = (chi[i+1, j]*(self.T[i+1, j]-self.T[i, j])/dr - chi[i, j]*(self.T[i, j]-self.T[i-1, j])/dr)/dr
                T_flux_z = (chi[i, j+1]*(self.T[i, j+1]-self.T[i, j])/dz - chi[i, j]*(self.T[i, j]-self.T[i, j-1])/dz)/dz
                heat_source = (P_alpha[i, j] - P_rad[i, j] + P_aux) * self.dt / (1.5 * self.n[i, j] * keV_to_J)
                T_new[i, j] = self.T[i, j] + self.dt * (T_flux_r + T_flux_z + heat_source)
        
        # Boundary conditions (zero gradient)
        n_new[0, :] = n_new[1, :]; n_new[-1, :] = n_new[-2, :]
        n_new[:, 0] = n_new[:, 1]; n_new[:, -1] = n_new[:, -2]
        T_new[0, :] = T_new[1, :]; T_new[-1, :] = T_new[-2, :]
        T_new[:, 0] = T_new[:, 1]; T_new[:, -1] = T_new[:, -2]
        
        # Numerical stability
        T_new = np.clip(T_new, 1e-3, 1e3)
        n_new = np.clip(n_new, 1e-10, 1e30)
        
        if not np.all(np.isfinite(T_new)) or not np.all(np.isfinite(n_new)):
            print("2D instability detected. Skipping update.")
            return
        
        # Update state
        self.n = n_new
        self.T = T_new
        self.time.append(self.time[-1] + self.dt)
        self.T_central.append(self.T[self.n_r//2, self.n_z//2])
        self.n_central.append(self.n[self.n_r//2, self.n_z//2])

    @property
    def R(self):
        return self.r

    @property
    def Z(self):
        return self.z

    def get_results(self):
        """Return current simulation results"""
        return {
            "T": self.T,
            "n": self.n,
            "R": self.r,
            "Z": self.z,
            "T_central": np.array(self.T_central),
            "n_central": np.array(self.n_central),
            "P_fusion_total": np.array(self.P_fusion_total),
            "E_fusion_total": np.array(self.E_fusion_total),
            "time": np.array(self.time)
        }

# ================================================
# STEP 4: SET UP GUI
# ================================================
def create_gui():
    """Create and configure the Tkinter GUI"""
    root = tk.Tk()
    root.title("Fusion Reactor Simulation")
    root.geometry("1100x800")
    root.minsize(900, 700)
    
    # Create notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Initialize simulations
    sims = {dev: LiveSim(params) for dev, params in devices.items()}
    sim2d_devices = {dev: LiveSim2D(params, n_r=32, n_z=32) for dev, params in devices.items()}
    
    # Assign colors to devices
    device_names = list(devices.keys())
    num_devices = len(device_names)
    cmap = cm.get_cmap('tab10' if num_devices <= 10 else 'tab20', num_devices)
    device_colors = {dev: cmap(i) for i, dev in enumerate(device_names)}
    
    # ================================================
    # TAB 1: Fusion Power Output
    # ================================================
    home_tab = ttk.Frame(notebook)
    notebook.add(home_tab, text="Home: Fusion Power Output")
    
    # Controls
    controls_frame = ttk.Frame(home_tab)
    controls_frame.pack(fill=tk.X, padx=10, pady=2)
    show_avg_var = tk.BooleanVar(value=True)
    log_scale_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(controls_frame, text="Show 10s Rolling Average", variable=show_avg_var).pack(side=tk.LEFT, padx=5)
    ttk.Checkbutton(controls_frame, text="Logarithmic Y-Axis", variable=log_scale_var).pack(side=tk.LEFT, padx=5)
    
    # Device summary table
    sticky_frame = ttk.Frame(home_tab)
    sticky_frame.pack(fill=tk.X, padx=10, pady=5)
    sticky_table = ttk.Treeview(sticky_frame, columns=("Device", "Current", "Avg10s", "TotalE"), show="headings", height=len(devices))
    sticky_table.heading("Device", text="Device")
    sticky_table.heading("Current", text="Current Power (W)")
    sticky_table.heading("Avg10s", text="Avg (10s) (W)")
    sticky_table.heading("TotalE", text="Total Energy (J)")
    sticky_table.column("Device", width=120, anchor="w")
    sticky_table.column("Current", width=120, anchor="e")
    sticky_table.column("Avg10s", width=120, anchor="e")
    sticky_table.column("TotalE", width=160, anchor="e")
    sticky_table.pack(fill=tk.X)
    for dev in devices:
        sticky_table.insert("", "end", iid=dev, values=(dev, "", "", ""))
    
    # Color table rows
    for dev in device_names:
        color = device_colors[dev]
        rgb = tuple(int(255 * c) for c in color[:3])
        hex_color = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
        sticky_table.tag_configure(dev, foreground=hex_color)
        sticky_table.item(dev, tags=(dev,))
    
    # Fusion power plot
    fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=100)
    lines_Pfusion = {}
    lines_Pfusion_avg = {}
    for dev in device_names:
        res = sims[dev].get_results()
        color = device_colors[dev]
        lines_Pfusion[dev], = ax3.plot(res["time"], res["P_fusion_total"], label=dev, alpha=0.4, linewidth=1, color=color)
        lines_Pfusion_avg[dev], = ax3.plot(res["time"], res["P_fusion_total"], label=f"{dev} (avg)", linewidth=2, color=color)
    ax3.set_title("Total Fusion Power Output vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Fusion Power (W)")
    ax3.set_yscale('log')
    ax3.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    handles, labels = ax3.get_legend_handles_labels()
    unique_labels = [l for l in labels if "(avg)" not in l]
    unique_handles = [h for h, l in zip(handles, labels) if "(avg)" not in l]
    ax3.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig3.tight_layout(rect=[0, 0, 0.85, 1])
    canvas3 = FigureCanvasTkAgg(fig3, master=home_tab)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=1)
    
    # ================================================
    # TAB 2: Central/Edge Evolution
    # ================================================
    evol_tab = ttk.Frame(notebook)
    notebook.add(evol_tab, text="Central/Edge Evolution")
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), dpi=100)
    lines_Tc, lines_Te, lines_nc, lines_ne = {}, {}, {}, {}
    for dev, sim in sims.items():
        res = sim.get_results()
        lines_Tc[dev], = axs[0].plot(res["time"], res["T_central"], label=f"{dev} (center)")
        lines_Te[dev], = axs[0].plot(res["time"], res["T_edge"], '--', label=f"{dev} (edge)")
        lines_nc[dev], = axs[1].plot(res["time"], res["n_central"], label=f"{dev} (center)")
        lines_ne[dev], = axs[1].plot(res["time"], res["n_edge"], '--', label=f"{dev} (edge)")
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
    
    # ================================================
    # TAB 3: Final Profiles
    # ================================================
    prof_tab = ttk.Frame(notebook)
    notebook.add(prof_tab, text="Final Profiles")
    
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    lines_Tprof, lines_nprof = {}, {}
    for dev, sim in sims.items():
        res = sim.get_results()
        lines_Tprof[dev], = axs2[0].plot(res["r"], res["T"], label=dev)
        lines_nprof[dev], = axs2[1].plot(res["r"], res["n"], label=dev)
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
    
    # ================================================
    # TAB 4: 2D Visualization
    # ================================================
    prof2d_tab = ttk.Frame(notebook)
    notebook.add(prof2d_tab, text="2D Profiles (All Devices)")
    
    ncols = 2
    nrows = len(device_names)
    fig2d, axs2d = plt.subplots(nrows, ncols, figsize=(10, 2.5 * nrows), dpi=100)
    if nrows == 1:
        axs2d = np.array([axs2d])  # Ensure 2D array
    
    imT2d = {}
    imn2d = {}
    cbarT2d = {}
    cbarn2d = {}
    
    for i, dev in enumerate(device_names):
        sim2d = sim2d_devices[dev]
        res2d = sim2d.get_results()
        # Temperature plot
        axT = axs2d[i, 0]
        imT2d[dev] = axT.imshow(res2d["T"].T, origin='lower', aspect='auto',
                               extent=[res2d["Z"][0], res2d["Z"][-1], res2d["R"][0], res2d["R"][-1]],
                               cmap='hot')
        cbarT2d[dev] = fig2d.colorbar(imT2d[dev], ax=axT)
        axT.set_title(f"{dev} - T (keV)")
        axT.set_xlabel("z (m)")
        axT.set_ylabel("r (m)")
        # Density plot
        axn = axs2d[i, 1]
        imn2d[dev] = axn.imshow(res2d["n"].T, origin='lower', aspect='auto',
                               extent=[res2d["Z"][0], res2d["Z"][-1], res2d["R"][0], res2d["R"][-1]],
                               cmap='viridis')
        cbarn2d[dev] = fig2d.colorbar(imn2d[dev], ax=axn)
        axn.set_title(f"{dev} - n (m^-3)")
        axn.set_xlabel("z (m)")
        axn.set_ylabel("r (m)")
    
    fig2d.tight_layout()
    canvas2d = FigureCanvasTkAgg(fig2d, master=prof2d_tab)
    canvas2d.draw()
    canvas2d.get_tk_widget().pack(fill=tk.BOTH, expand=1)
    
    # ================================================
    # LIVE UPDATE FUNCTION
    # ================================================
    def live_update():
        all_powers = []
        window_seconds = 30  # Display last 30 seconds
        
        # Update 1D simulations and plots
        for dev, sim in sims.items():
            # Advance simulation
            for _ in range(5):
                sim.step()
            res = sim.get_results()
            
            # Update fusion power plot
            t = res["time"]
            idx_start = max(0, len(t) - int(window_seconds / sim.dt))
            t_win = t[idx_start:]
            pf_win = res["P_fusion_total"][idx_start:]
            
            # Compute 10-second rolling average
            n_last = int(10 / sim.dt)
            if len(pf_win) >= n_last:
                pf_avg = np.convolve(pf_win, np.ones(n_last)/n_last, mode='valid')
                t_avg = t_win[n_last-1:]
            else:
                pf_avg = np.array([np.mean(pf_win)]) if pf_win.size > 0 else np.array([0])
                t_avg = t_win[-len(pf_avg):] if t_win.size > 0 else np.array([0])
            
            lines_Pfusion[dev].set_data(t_win, pf_win)
            if show_avg_var.get():
                lines_Pfusion_avg[dev].set_data(t_avg, pf_avg)
                lines_Pfusion_avg[dev].set_visible(True)
            else:
                lines_Pfusion_avg[dev].set_visible(False)
            
            all_powers.extend(pf_win)
            all_powers.extend(pf_avg)
            
            # Update table
            n_last = int(10 / sim.dt)
            if len(res["P_fusion_total"]) > n_last:
                avg_power = np.mean(res["P_fusion_total"][-n_last:])
            else:
                avg_power = np.mean(res["P_fusion_total"])
            total_energy = res["E_fusion_total"][-1]
            current_power = res["P_fusion_total"][-1]
            sticky_table.set(dev, "Current", f"{current_power:.3e}")
            sticky_table.set(dev, "Avg10s", f"{avg_power:.3e}")
            sticky_table.set(dev, "TotalE", f"{total_energy:.3e}")
        
        # Update 2D simulations and plots
        for dev in device_names:
            sim2d = sim2d_devices[dev]
            for _ in range(2):  # Advance slower for performance
                sim2d.step()
            res2d = sim2d.get_results()
            imT2d[dev].set_data(res2d["T"].T)
            imT2d[dev].set_clim(np.min(res2d["T"]), np.max(res2d["T"]))
            imn2d[dev].set_data(res2d["n"].T)
            imn2d[dev].set_clim(np.min(res2d["n"]), np.max(res2d["n"]))
        
        # Adjust power plot scaling
        if all_powers:
            max_power = max([p for p in all_powers if p > 0] or [1])
            if log_scale_var.get():
                ax3.set_yscale('log')
                ax3.set_ylim(1e-2, max_power * 1.1)
            else:
                ax3.set_yscale('linear')
                ax3.set_ylim(0, max_power * 1.1)
        
        # Update all plots
        axs[0].relim(); axs[0].autoscale_view()
        axs[1].relim(); axs[1].autoscale_view()
        axs2[0].relim(); axs2[0].autoscale_view()
        axs2[1].relim(); axs2[1].autoscale_view()
        ax3.relim(); ax3.autoscale_view()
        canvas.draw()
        canvas2.draw()
        canvas3.draw()
        canvas2d.draw()
        
        # Schedule next update
        root.after(100, live_update)
    
    # Start live updates
    live_update()
    
    return root

# ================================================
# STEP 5: RUN THE APPLICATION
# ================================================
if __name__ == "__main__":
    # Configure numpy error handling
    np.seterr(divide='raise', invalid='raise', over='raise', under='ignore')
    
    # Create and run GUI
    app = create_gui()
    app.mainloop()
