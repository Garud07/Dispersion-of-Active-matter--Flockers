# 2d_active_matter
# Simulation12_S&S_Start_Fix_Pt_Rint=0.05

This repository contains the twelfth stage of the *Dispersion of Active Matter in Turbulence* study.  
It continues the **no-flow Stokes + Self (S&S)** configuration used in Simulations 10 and 11,  
but with an **interaction radius of `Rint = 0.05`**, allowing moderately strong coupling between neighboring particles.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** None  
- **Initial Positions:** All particles start from a single fixed point  
- **Dynamics:**  
  - **Self-propulsion:** Active velocity with stochastic orientation  
  - **Stokes drag:** Linear damping due to viscous resistance  
- **Interaction Radius (`Rint`):** 0.05 (moderate local alignment)  
- **Objective:**  
  Explore the effect of moderate interaction range on the evolution of particle clusters, order parameter, and spreading rate.  
  This case bridges the behavior observed between `Rint = 1` (strong coupling) and `Rint = 0.03` (weak coupling).

---

## 🧠 How to Run

```bash
python main.py
