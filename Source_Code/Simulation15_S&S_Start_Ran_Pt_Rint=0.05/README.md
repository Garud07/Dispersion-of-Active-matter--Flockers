# 2d_active_matter
# Simulation15_S&S_Start_Ran_Pt_Rint=0.05

This repository contains the fifteenth stage of the *Dispersion of Active Matter in Turbulence* project.  
It investigates **Stokes + Self (S&S)** particle dynamics under **no-flow conditions** with  
**random initial positions** and a **small interaction radius (`Rint = 0.05`)**.  

This configuration demonstrates how reducing the interaction range leads to loss of large-scale coherence and the emergence of localized or independent motion.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** None  
- **Initial Positions:** Random and uniformly distributed  
- **Dynamics:**  
  - **Self-propulsion:** Constant active velocity with noise  
  - **Stokes drag:** Linear damping from viscous effects  
- **Interaction Radius (`Rint`):** 0.05 (very local alignment)  
- **Objective:**  
  Analyze how limited-range interactions affect clustering, order formation, and dispersion rate  
  compared to higher `Rint` cases (0.3, 1.0).

---

## 🧠 How to Run

```bash
python main.py
