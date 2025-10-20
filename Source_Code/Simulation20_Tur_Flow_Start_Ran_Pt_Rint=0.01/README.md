# 2d_active_matter
# Simulation20_Tur_Flow_Start_Ran_Pt_Rint=0.01

This repository contains the twentieth stage of the *Dispersion of Active Matter in Turbulence* study.  
It follows the same procedure as Simulation19 — turbulence is first developed **without particles** for ~200 s —  
but here, particles are injected with a **smaller interaction radius (`Rint = 0.01`)**,  
making inter-particle coupling very weak compared to hydrodynamic (turbulent) effects.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** 2D forced turbulence (periodic domain)  
- **Flow Initialization:** Turbulence evolved without particles for ~200 s  
- **Particle Injection:** After turbulence reaches steady-state  
- **Particle Initialization:** Random and uniform  
- **Dynamics (post-injection):**
  - **Self-propulsion:** Active velocity with stochastic direction  
  - **Stokes drag:** Linear viscous damping  
  - **Advection:** Particle motion coupled to turbulent velocity field  
- **Interaction Radius (`Rint`):** 0.01 (weak interaction regime)  
- **Objective:**  
  Study how weakly interacting self-propelled particles behave in pre-developed turbulence  
  and quantify the balance between random flow dispersion and limited alignment dynamics.

---

## 🧠 How to Run

```bash
python main.py
