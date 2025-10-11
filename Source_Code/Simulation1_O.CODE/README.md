# 2d_active_matter
# Simulation 1 – Base Turbulent Flow

This simulation generates a 2-D turbulent velocity field using **Dedalus**,  
with stochastic forcing applied in the wavenumber band *k ∈ [3, 4]*.

---

## 📋 Description
- Governing equations: vorticity–streamfunction formulation  
- Time integration: RK222 scheme  
- Forcing strategy: power-controlled random forcing  
- Outputs:
  - `snapshots/` → instantaneous vorticity & streamfunction  
  - `spectra/` → energy / enstrophy spectra  
  - `timeseries/` → kinetic energy & enstrophy vs time  

---

## ▶️ Run Command
```bash
mpirun -n 4 python main.py
