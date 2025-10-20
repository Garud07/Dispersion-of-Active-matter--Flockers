# 2d_active_matter
# Simulation16_S&SIT_Start_Ran_Pt_Rint=0.5

This repository contains the sixteenth stage of the *Dispersion of Active Matter in Turbulence* project.  
This case introduces **fluid turbulence** into the Stokes + Self-propelled (S&S) framework — referred to as **S&SIT (Stokes and Self in Turbulence)**.  
Particles are **randomly initialized** and interact with a **moderate interaction radius (`Rint = 0.5`)**.  

The goal is to study how active particles behave when subjected to both **their intrinsic dynamics** and the **external turbulent flow field**.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** Turbulent (2D periodic domain)  
- **Flow Type:** Forced turbulence with steady-state energy injection  
- **Particle Initialization:** Random, uniformly distributed  
- **Dynamics:**  
  - **Self-propulsion:** Active velocity term with directional noise  
  - **Stokes drag:** Linear damping coupling particle to the local fluid velocity  
  - **Advection:** Particle position updated using local fluid velocity interpolated from the flow field  
- **Interaction Radius (`Rint`):** 0.5 (moderate-range alignment)  
- **Objective:**  
  Investigate how turbulence alters clustering, dispersion, and order in active particle systems  
  compared to no-flow scenarios.

---

## 🧠 How to Run

```bash
python main.py
