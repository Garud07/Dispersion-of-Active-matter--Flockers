# 2d_active_matter
# Simulation10_S&S_Start_Fix_Pt_Rint=1

This repository contains the tenth stage of the *Dispersion of Active Matter in Turbulence* study.  
The setup explores **self-propelled particles with Stokesian drag (S&S)** under **no-flow conditions**,  
but unlike previous random initializations, **all particles start from a fixed location**.  
The parameter `Rint = 1` (interaction radius) controls the local alignment or interaction strength among particles.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** None  
- **Initial Positions:** All particles start from a fixed point  
- **Dynamics:**  
  - **Self-propulsion:** Constant active velocity with stochastic direction  
  - **Stokes drag:** Linear damping term  
- **Interaction Radius (`Rint`):** 1 (defines neighbour interaction zone)  
- **Objective:**  
  Examine how particle dispersion and order emerge from a concentrated start when both self-propulsion and viscous drag are active.

---

## 🧠 How to Run

```bash
python main.py
