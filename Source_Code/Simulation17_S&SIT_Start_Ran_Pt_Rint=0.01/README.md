# 2d_active_matter
# Simulation17_S&SIT_Start_Ran_Pt_Rint=0.01

This repository contains the seventeenth stage of the *Dispersion of Active Matter in Turbulence* study.  
It extends the **S&SIT (Stokes and Self in Turbulence)** framework from Simulation16 by reducing the **interaction radius to `Rint = 0.01`**,  
making this a **weakly interacting** or **nearly independent** particle system.  

Particles are advected by a turbulent flow while self-propelling and experiencing Stokes drag,  
but with minimal alignment or neighbor coupling.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** Turbulent (2D periodic domain)  
- **Flow Generation:** Spectral forcing to maintain statistically steady turbulence  
- **Initial Positions:** Random and uniformly distributed  
- **Dynamics:**  
  - **Self-propulsion:** Constant speed with small orientation noise  
  - **Stokes drag:** Linear viscous coupling to the local flow velocity  
  - **Advection:** Motion governed by both self-propulsion and turbulent velocity field  
- **Interaction Radius (`Rint`):** 0.01 (minimal coupling)  
- **Objective:**  
  Study how nearly non-interacting active particles respond to turbulence,  
  comparing dispersion, clustering, and alignment with higher `Rint` cases.

---

## 🧠 How to Run

```bash
python main.py
