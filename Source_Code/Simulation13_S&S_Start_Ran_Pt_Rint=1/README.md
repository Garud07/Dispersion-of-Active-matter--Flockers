# 2d_active_matter
# Simulation13_S&S_Start_Ran_Pt_Rint=1

This repository contains the thirteenth stage of the *Dispersion of Active Matter in Turbulence* study.  
It investigates the **Stokes + Self (S&S)** dynamics under **no-flow conditions**,  
with **random initial particle positions** and a **large interaction radius (`Rint = 1`)**.  

This configuration explores how collective order emerges from spatially distributed particles that strongly influence one another through alignment and drag coupling.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** None  
- **Initial Positions:** Random and uniform across domain  
- **Dynamics:**  
  - **Self-propulsion:** Constant active velocity with orientation noise  
  - **Stokes drag:** Linear viscous damping  
- **Interaction Radius (`Rint`):** 1 (strong, long-range interactions)  
- **Objective:**  
  Study the development of large-scale alignment and clustering when particles start randomly but can interact strongly with distant neighbors.

---

## 🧠 How to Run

```bash
python main.py
