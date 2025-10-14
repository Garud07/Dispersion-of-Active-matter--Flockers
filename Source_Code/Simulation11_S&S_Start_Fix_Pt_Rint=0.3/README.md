# 2d_active_matter
# Simulation11_S&S_Start_Fix_Pt_Rint=0.03

This repository contains the eleventh stage of the *Dispersion of Active Matter in Turbulence* study.  
It continues the **no-flow, Stokes + Self (S&S)** investigation from Simulation 10,  
but with a **smaller interaction radius (`Rint = 0.03`)**, resulting in weaker or more localized particle coupling.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** None  
- **Initial Positions:** All particles initialized at a fixed point  
- **Dynamics:**  
  - **Self-propulsion:** Active velocity with orientation noise  
  - **Stokes drag:** Linear viscous damping  
- **Interaction Radius (`Rint`):** 0.03 (very localized interactions)  
- **Objective:**  
  Study the impact of decreasing interaction radius on clustering, alignment, and dispersion.  
  This helps identify the threshold beyond which particles behave independently rather than as a collective.

---

## 🧠 How to Run

```bash
python main.py
