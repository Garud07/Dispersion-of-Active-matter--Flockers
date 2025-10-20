# 2d_active_matter
# Simulation18_S&SIT_Start_Ran_Pt_Rint=0.001

This repository contains the eighteenth stage of the *Dispersion of Active Matter in Turbulence* study.  
In this setup, **Stokes + Self-propelled (S&S)** particles evolve **in a turbulent flow field (S&SIT)**  
with **random initial positions** and an **extremely small interaction radius (`Rint = 0.001`)**.  

This case represents the **non-interacting limit**, where particle–particle alignment is negligible,  
and motion is dominated purely by **turbulent advection**, **viscous drag**, and **self-propulsion**.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** Turbulent (2D periodic domain)  
- **Flow Forcing:** Spectral band-limited forcing to sustain turbulence  
- **Particle Initialization:** Random and uniform  
- **Dynamics:**  
  - **Self-propulsion:** Constant active speed with stochastic noise  
  - **Stokes drag:** Linear damping term coupling with local flow velocity  
  - **Advection:** Passive transport by turbulent velocity field  
- **Interaction Radius (`Rint`):** 0.001 (non-interacting limit)  
- **Objective:**  
  Simulate and analyze the dispersion behavior of almost independent active particles in turbulence,  
  where hydrodynamic and self-propulsion effects dominate over alignment interactions.

---

## 🧠 How to Run

```bash
python main.py
