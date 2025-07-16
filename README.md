# 🔥 Fusion Reactor Simulation GUI

This project is a real-time interactive simulation and visualization tool for comparing the performance of different fusion reactor types using both 1D and 2D transport models.

Built with:
- **Python**
- **Tkinter** for the GUI
- **Matplotlib** for dynamic plotting
- **NumPy** for efficient numerical computation

---

## 📸 Preview

> 💡 Add screenshots here after running the app!
```bash
# Example:
![Fusion Power Output Tab](screenshots/fusion_power_output.png)
![2D Profiles Tab](screenshots/2d_profiles.png)
```

---

## 🔬 Simulated Reactor Types

- Tokamak
- ITER
- Polaris (Helion Energy)
- LDX Junior (Levitated Dipole)
- Stellarator
- Z-Pinch
- ICF (Inertial Confinement Fusion)

Each reactor is initialized with realistic physics parameters including:
- Fuel type (DT, DD, DHe3)
- Plasma volume and radius
- Impurity species, charge, and radiation losses
- Magnetic field strength and current

---

## ⚙️ Features

- 🔁 Real-time 1D and 2D plasma transport simulation
- 📊 Live plots of:
  - Fusion power over time (with 10s rolling average)
  - Central and edge temperature/density
  - Final radial profiles
  - 2D temperature and density heatmaps for each reactor
- 🧪 Physical models include:
  - Gyro-Bohm diffusion
  - Bremsstrahlung and line radiation
  - Alpha heating and auxiliary power injection
  - Fusion energy cross-sections for different fuels

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.8+
- Packages:
  - `numpy`
  - `matplotlib`
  - (comes with `tkinter` by default in most Python installations)

### 📦 Installation

```bash
git clone https://github.com/yourusername/fusion-reactor-sim.git
cd fusion-reactor-sim
pip install -r requirements.txt
python main.py
```

If you’re using a system without `tkinter`, install it via:

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**macOS (with Homebrew):**
```bash
brew install python-tk
```

---

## 📁 File Structure

```
fusion-reactor-sim/
├── main.py                # Full application entry point
├── README.md              # This file
├── requirements.txt       # List of Python dependencies
├── /screenshots           # (Optional) Images for README preview
```

---

## 📈 Physics Model Overview

### 🔥 Fusion Power

Calculated using fuel-specific cross-section approximations:
- DT: σv ≈ 3.68 × 10⁻¹⁸ (T / 17.6)^0.3 e^{-(T/69)^0.8}

Total fusion power:
P_fusion = n₁ n₂ · σv · E_fusion

### 🌡️ Energy Balance

3/2 ∂(nT)/∂t = ∇ · (χ ∇T) + P_α + P_aux - P_rad

Radiation includes:
- Bremsstrahlung: P_brem ∝ Z_eff n² √T
- Line radiation: P_line ∝ Z_imp² · frac · n² e^{-T/10}

---

## 📊 GUI Tabs

1. **Home**: Real-time fusion power output graph and summary table
2. **Central/Edge Evolution**: Tracks changes in core and edge plasma
3. **Final Profiles**: Final temperature and density radial distributions
4. **2D Profiles**: R-Z heatmaps of temperature and density per reactor

---

## 🤝 Credits

Developed by Jin-Zhao, Ayaan, Lucas  
For simulation accuracy and physics formulation, references include:
- ITER Technical Design Report
- Helion Energy & NIF research materials
- Principles of Plasma Physics (Chen, 2016)

---

## 📝 License

MIT License.  
Feel free to modify, redistribute, or use for academic purposes.
