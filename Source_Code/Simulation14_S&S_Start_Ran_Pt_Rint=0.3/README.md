# 2d_active_matter
# Simulation14_S&S_Start_Ran_Pt_Rint=0.3

This repository contains the fourteenth stage of the *Dispersion of Active Matter in Turbulence* study.  
It explores **Stokes + Self (S&S)** dynamics under **no-flow conditions**,  
where particles are **randomly initialized** and interact with a **moderate interaction radius (`Rint = 0.3`)**.  

This case bridges the transition between strong long-range alignment (Rint = 1) and localized interactions (Rint ≪ 1).

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** None  
- **Initial Positions:** Random and uniformly distributed  
- **Dynamics:**  
  - **Self-propulsion:** Active velocity with directional noise  
  - **Stokes drag:** Linear viscous damping  
- **Interaction Radius (`Rint`):** 0.3 (moderate local alignment)  
- **Objective:**  
  Analyze how reduced interaction range affects collective motion, clustering, and dispersion  
  when starting from random spatial configurations.

---

## 🧠 How to Run

```bash
python main.py
