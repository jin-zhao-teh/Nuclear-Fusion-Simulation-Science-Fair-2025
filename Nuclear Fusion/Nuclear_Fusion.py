import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk

try:
    import mplcursors  # type: ignore
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False

# --- CONSTANTS ---
kB = 1.602e-16  # J/keV
sigma_SB = 5.670374419e-8  # W/m^2/K^4
mu0 = 4 * np.pi * 1e-7  # H/m
e = 1.602e-19  # C

# --- DEVICE PARAMETERS ---
devices = {
    "Tokamak": {
        "n0": 5e19, "T0": 10, "V": 100, "fuel": "DT", "I_p": 1e6, "B0": 5, "R": 2, "a": 0.5,
        "coil_L": 1e-6, "coil_thickness": 0.1, "vessel_V": 200, "A_wall": 100, "T_cold": 4, "T_hot": 300,
        "emissivity": 0.2, "k_strut": 0.1, "A_strut": 0.01, "L_strut": 1, "eff_fridge": 0.3,
        "Z_eff": 1.5, "impurity_frac": 0.01, "alpha_loss_frac": 0.1, "kappa": 1.7,
        "D": 1.0, "T_edge": 0.1, "P_aux_max": 1e7, "Kp": 1e5, "Ki": 1e-2, "Kd": 1e3
    },
    "ITER": {
        "n0": 1e20, "T0": 20, "V": 840, "fuel": "DT", "I_p": 1.5e6, "B0": 5.3, "R": 6.2, "a": 2,
        "coil_L": 2e-6, "coil_thickness": 0.2, "vessel_V": 2000, "A_wall": 400, "T_cold": 4, "T_hot": 300,
        "emissivity": 0.2, "k_strut": 0.1, "A_strut": 0.05, "L_strut": 2, "eff_fridge": 0.3,
        "Z_eff": 1.5, "impurity_frac": 0.01, "alpha_loss_frac": 0.05, "kappa": 1.7,
        "D": 1.0, "T_edge": 0.1, "P_aux_max": 5e7, "Kp": 1e5, "Ki": 1e-2, "Kd": 1e3
    },
    "Polaris": {
        "n0": 5e22, "T0": 20, "V": 0.5, "fuel": "DHe3", "I_p": 0, "B0": 10, "R": 0.5, "a": 0.1,
        "coil_L": 1e-7, "coil_thickness": 0.05, "vessel_V": 1, "A_wall": 2, "T_cold": 20, "T_hot": 300,
        "emissivity": 0.1, "k_strut": 0.2, "A_strut": 0.005, "L_strut": 0.2, "eff_fridge": 0.1,
        "Z_eff": 2.0, "impurity_frac": 0.02, "alpha_loss_frac": 0.2, "kappa": 1.0,
        "D": 0.5, "T_edge": 0.2, "P_aux_max": 1e6, "Kp": 1e4, "Ki": 1e-3, "Kd": 1e2
    },
    "LDX Junior": {
        "n0": 1e19, "T0": 1, "V": 0.014, "fuel": "DD", "I_p": 0, "B0": 0.5, "R": 0.2, "a": 0.05,
        "coil_L": 1e-7, "coil_thickness": 0.02, "vessel_V": 0.1, "A_wall": 0.5, "T_cold": 20, "T_hot": 300,
        "emissivity": 0.1, "k_strut": 0.2, "A_strut": 0.001, "L_strut": 0.1, "eff_fridge": 0.1,
        "Z_eff": 1.2, "impurity_frac": 0.005, "alpha_loss_frac": 0.3, "kappa": 1.0,
        "D": 0.1, "T_edge": 0.05, "P_aux_max": 1e5, "Kp": 1e3, "Ki": 1e-4, "Kd": 1e1
    },
    "Stellarator": {
        "n0": 2e19, "T0": 8, "V": 30, "fuel": "DT", "I_p": 0, "B0": 3, "R": 1.5, "a": 0.4,
        "coil_L": 1e-6, "coil_thickness": 0.08, "vessel_V": 60, "A_wall": 20, "T_cold": 4, "T_hot": 300,
        "emissivity": 0.2, "k_strut": 0.1, "A_strut": 0.005, "L_strut": 0.5, "eff_fridge": 0.3,
        "Z_eff": 1.3, "impurity_frac": 0.008, "alpha_loss_frac": 0.1, "kappa": 1.7
    },
    "Z-pinch": {
        "n0": 1e21, "T0": 5, "V": 0.01, "fuel": "DT", "I_p": 2e6, "B0": 20, "R": 0.01, "a": 0.005,
        "coil_L": 1e-8, "coil_thickness": 0.01, "vessel_V": 0.02, "A_wall": 0.1, "T_cold": 20, "T_hot": 300,
        "emissivity": 0.1, "k_strut": 0.2, "A_strut": 0.0005, "L_strut": 0.01, "eff_fridge": 0.1,
        "Z_eff": 1.7, "impurity_frac": 0.03, "alpha_loss_frac": 0.2, "kappa": 1.0
    },
    "ICF": {
        "n0": 1e26, "T0": 10, "V": 1e-9, "fuel": "DT", "I_p": 0, "B0": 0, "R": 0.001, "a": 0.0005,
        "coil_L": 0, "coil_thickness": 0, "vessel_V": 1e-8, "A_wall": 1e-6, "T_cold": 20, "T_hot": 300,
        "emissivity": 0.1, "k_strut": 0.2, "A_strut": 1e-8, "L_strut": 1e-4, "eff_fridge": 0.1,
        "Z_eff": 1.1, "impurity_frac": 0.001, "alpha_loss_frac": 0.01, "kappa": 1.0
    }
}

# --- PHYSICS MODULES ---

def tau_E_ITER98y2(n20, P_tot_MW, R, a, B0, kappa=1.7, M=2.5):
    """ITER-98(y,2) energy confinement time scaling law for tokamaks."""
    return 0.0562 * (n20**0.41) * (P_tot_MW**-0.69) * (R**1.97) * (a**0.58) * (B0**0.15) * (kappa**0.78) * (M**0.19)

def sigma_v(T, fuel):
    """Bosch-Hale fit for reactivity <σv> in m^3/s. T in keV."""
    if fuel == "DT":
        return 3.68e-18 * (T / 17.6)**(0.3) * np.exp(-((T / 69)**0.8))
    elif fuel == "DHe3":
        return 2.5e-21 * T**(-2/3) * np.exp(-18.8 * T**(-1/3))
    elif fuel == "DD":
        return 1.6e-22 * T**(-2/3) * np.exp(-18.76 * T**(-1/3))
    else:
        return 0.0

def fusion_energy(fuel):
    """Energy per reaction in Joules."""
    if fuel == "DT": return 17.6e6 * e
    elif fuel == "DHe3": return 18.3e6 * e
    elif fuel == "DD": return 3.65e6 * e
    else: return 0.0

def compute_bremsstrahlung(n_e, T_keV, V, Z_eff=1.0):
    """Relativistic Bremsstrahlung loss (W)."""
    return 1.69e-38 * Z_eff * n_e**2 * T_keV**0.5 * V

def compute_impurity_radiation(n, T, V, Z_eff, impurity_frac):
    """Estimate impurity radiation loss (W)."""
    C_imp = 1e-38  # Placeholder, replace with OPEN-ADAS for high fidelity
    return C_imp * Z_eff**3 * impurity_frac * n**2 * V

def cooling_function(T, Z_imp):
    """Placeholder for OPEN-ADAS cooling function."""
    return 1e-34 * (Z_imp**2) * np.exp(-T/10)

def compute_synchrotron(n_e, T_keV, B, R, a, alpha_r=0.1):
    """Trubnikov synchrotron loss (W)."""
    tau = 5.0 * a * T_keV**2 * B / (R * n_e)
    return 6.21e-34 * T_keV**4.5 * B**2 * R * a * n_e * (1 - np.exp(-tau)) / (1 - alpha_r * np.exp(-tau))

def eta_spitzer(T):
    """Spitzer resistivity (Ohm*m) for fully ionized plasma, T in keV."""
    Z = 1
    ln_Lambda = 17
    return 1.65e-9 * Z * ln_Lambda / (T**(3/2))

def compute_ohmic_heating(T, I_p):
    """Ohmic heating (W) for tokamaks/stellarators."""
    eta = eta_spitzer(T)
    return eta * I_p**2

def compute_tau_E(device, params, n, T, P_heat, B, R, a):
    """Confinement time (s) using scaling laws."""
    if device in ["Tokamak", "ITER"]:
        n20 = n / 1e20
        P_MW = max(P_heat / 1e6, 1e-3)
        return tau_E_ITER98y2(n20, P_MW, R, a, B, params.get('kappa', 1.7))
    elif device == "Stellarator":
        # ISS04 scaling (placeholder)
        return params.get("tau_E", 0.5)
    elif device in ["Z-pinch", "Polaris"]:
        m_i = 2 * 1.67e-27  # D-T ion mass
        v_A = B / np.sqrt(mu0 * n * m_i)
        return R / v_A
    elif device == "ICF":
        return 1e-9
    else:
        return params.get("tau_E", 1.0)

def compute_radiation(n_e, T, V, B, R, a, f_imp, Z_imp, Z_eff, alpha_r=0.1):
    P_brem = compute_bremsstrahlung(n_e, T, V, Z_eff)
    P_synch = compute_synchrotron(n_e, T, B, R, a, alpha_r)
    P_line = n_e**2 * f_imp * cooling_function(T, Z_imp) * V
    return P_brem + P_synch + P_line

def check_stability(device, params, n, T, I_p, B):
    if device == "Tokamak":
        beta = (2 * mu0 * n * kB * T) / B**2
        beta_lim = 0.028 * I_p / (params['a'] * B)
        if beta > beta_lim:
            return "Disrupt: Beta limit"
        n_G = (I_p / 1e6) / (np.pi * params['a']**2) * 1e20
        if n > n_G:
            return "Disrupt: Density limit"
        q = 5 * params['a']**2 * B / (params['R'] * I_p)
        if q < 2:
            return "Disrupt: Kink instability"
    return "Stable"

def compute_plasma_step(device, params, state, dt):
    """Advance plasma state by one time step."""
    n, T, W = state["n"], state["T"], state["W"]
    fuel = params["fuel"]
    V = params["V"]
    I_p = params["I_p"]
    B = params["B0"]
    R = params["R"]
    a = params["a"]
    Z_eff = params["Z_eff"]
    impurity_frac = params["impurity_frac"]
    alpha_loss_frac = params["alpha_loss_frac"]
    kappa = params.get("kappa", 1.7)

    # Reactivity and fusion power
    sv = sigma_v(T, fuel)
    E_fus = fusion_energy(fuel)
    if fuel == "DT":
        n_D = n_T = n / 2
        P_fusion = n_D * n_T * sv * E_fus * V
        f_alpha = 0.2
    else:
        P_fusion = 0.25 * n**2 * sv * E_fus * V
        f_alpha = 0.0
    P_alpha = f_alpha * P_fusion * (1 - alpha_loss_frac)

    # Ohmic heating
    P_ohm = compute_ohmic_heating(T, I_p) if I_p > 0 else 0.0
    P_aux = 0.0  # No auxiliary heating in this demo

    # Confinement time
    P_heat = P_alpha + P_ohm + P_aux
    tau_E = compute_tau_E(device, params, n, T, P_heat, B, R, a)

    # Losses
    P_cond = 3 * n * kB * T * V / tau_E
    P_rad_brem = compute_bremsstrahlung(n, T, V, Z_eff)
    P_rad_imp = compute_impurity_radiation(n, T, V, Z_eff, impurity_frac)
    P_rad = P_rad_brem + P_rad_imp
    P_synch = compute_synchrotron(n, T, B, R, a)
    P_loss = P_cond + P_rad + P_synch

    # Energy balance
    dW_dt = P_alpha + P_ohm + P_aux - P_loss
    W_new = W + dW_dt * dt
    T_new = max(W_new / (3 * n * kB * V), 0.01)

    # Fuel burn-up (for long pulses)
    if fuel == "DT":
        dn_dt = -0.5 * (n/2) * (n/2) * sv
        n_new = n + dn_dt * dt
    else:
        n_new = n

    # Q and net power
    Q = P_fusion / (P_ohm + P_aux) if (P_ohm + P_aux) > 0 else 0.0
    P_net = P_fusion - P_loss

    return {
        "n": n_new, "T": T_new, "W": W_new, "P_fusion": P_fusion, "P_alpha": P_alpha,
        "P_ohm": P_ohm, "P_cond": P_cond, "P_rad": P_rad, "P_synch": P_synch,
        "Q": Q, "P_net": P_net, "tau_E": tau_E
    }

def compute_cryo_losses(params):
    """Compute total cryogenic heat load and wall-plug power."""
    T_avg = (params["T_hot"] + params["T_cold"]) / 2
    k_strut = params["k_strut"] * (1 + 0.001 * T_avg)
    Q_cond = k_strut * params["A_strut"] / params["L_strut"] * (params["T_hot"] - params["T_cold"])
    Q_rad = params["emissivity"] * sigma_SB * params["A_wall"] * (params["T_hot"]**4 - params["T_cold"]**4)
    Q_total = Q_cond + Q_rad
    COP_Carnot = params["T_cold"] / (params["T_hot"] - params["T_cold"])
    COP_real = params["eff_fridge"] * COP_Carnot
    wallplug_power = Q_total / COP_real if COP_real > 0 else 0.0
    return Q_cond, Q_rad, Q_total, COP_real, wallplug_power

def compute_magnetics(params, I_p):
    """Compute magnetic field, stored energy, and coil stress."""
    B = params["B0"]
    R = params["R"]
    L = params["coil_L"]
    thickness = params["coil_thickness"]
    E_mag = 0.5 * L * I_p**2
    hoop_stress = B**2 * R / (mu0 * thickness) if thickness > 0 else 0.0
    return B, E_mag, hoop_stress

def compute_vacuum(params, p, Q_load=1e-6, S_pump=1.0):
    """Vacuum pressure evolution (Pa)."""
    V_vessel = params["vessel_V"]
    dp_dt = (Q_load - S_pump * p) / V_vessel
    return p + dp_dt * dt

def compute_structural(params, T, T0):
    """Thermal expansion and stress."""
    alpha = 1e-5  # 1/K
    L0 = params["L_strut"]
    Delta_L = alpha * L0 * (T - T0)
    return Delta_L

def compute_diagnostics(device, state):
    """Stub for diagnostics (e.g., Mirnov coil, neutron yield)."""
    if device == "ICF":
        return state["P_fusion"] * 1e-8
    return 0.0

# --- MAIN SIMULATION LOOP ---
dt = 0.01
t_max = 10.0
time = np.arange(0, t_max + dt, dt)

results = {dev: {"T": [], "P_fusion": [], "Q": [], "P_net": [], "wallplug": [], "coil_stress": [], "pressure": [], "yield": []} for dev in devices}
final_states = {}

for dev, params in devices.items():
    n = params["n0"]
    T = params["T0"]
    V = params["V"]
    W = 3 * n * (T * kB) * V
    I_p = params["I_p"]
    p = 1e-3  # Initial vacuum pressure (Pa)
    state = {"n": n, "T": T, "W": W}
    for t in time:
        state = compute_plasma_step(dev, params, state, dt)
        _, _, _, _, wallplug = compute_cryo_losses(params)
        _, _, coil_stress = compute_magnetics(params, I_p)
        p = compute_vacuum(params, p)
        diag = compute_diagnostics(dev, state)
        results[dev]["T"].append(state["T"])
        results[dev]["P_fusion"].append(state["P_fusion"])
        results[dev]["Q"].append(state["Q"])
        results[dev]["P_net"].append(state["P_net"])
        results[dev]["wallplug"].append(wallplug)
        results[dev]["coil_stress"].append(coil_stress)
        results[dev]["pressure"].append(p)
        results[dev]["yield"].append(diag)
    final_states[dev] = {
        "T_final": state["T"], "P_fusion_final": state["P_fusion"], "Q_final": state["Q"],
        "P_net_final": state["P_net"], "wallplug_final": wallplug, "coil_stress_final": coil_stress,
        "pressure_final": p, "yield_final": diag
    }

# --- VISUALIZATION ---
import matplotlib.gridspec as gridspec

fig = plt.figure(constrained_layout=True, figsize=(18, 16))
gs = gridspec.GridSpec(4, 2, figure=fig)

plot_list = [
    ("T", "Plasma Temperature (keV)"),
    ("P_fusion", "Fusion Power (W)"),
    ("Q", "Fusion Gain Q"),
    ("P_net", "Net Power (W)"),
    ("wallplug", "Refrigerator Wallplug Power (W)"),
    ("coil_stress", "Coil Hoop Stress (Pa)"),
    ("pressure", "Vessel Pressure (Pa)"),
    ("yield", "ICF Neutron Yield (arb)")
]

for i, (key, label) in enumerate(plot_list):
    ax = fig.add_subplot(gs[i//2, i%2])
    for dev in devices:
        ax.plot(time, results[dev][key], label=dev)
    ax.set_title(label)
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle='--', alpha=0.6)
    if i == 0:
        ax.legend(fontsize=8)
    if MPLCURSORS_AVAILABLE:
        mplcursors.cursor(ax.get_lines(), hover=True)

plt.tight_layout()
plt.show()

# --- SUMMARY TABLE ---
df = pd.DataFrame(final_states).T
print(df)