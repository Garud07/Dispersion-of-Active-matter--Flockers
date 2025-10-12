# 2d_active_matter
# Simulation8_Both_Stokes&Self

This repository contains the eighth stage of the *Dispersion of Active Matter in Turbulence* project.  
In this simulation, particles experience **both self-propulsion and Stokesian drag**, but **no external flow field**.  
The goal is to examine how the competition between **active drive** and **viscous damping** shapes the system’s collective dynamics and dispersion.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** None  
- **Initial Distribution:** Random and uniform across domain  
- **Dynamics:** Combination of  
  - **Self-propulsion:** Active velocity with noise  
  - **Stokes drag:** Linear damping proportional to particle velocity  
- **Noise:** Stochastic perturbation in direction or velocity  
- **Objective:**  
  Investigate how self-propelled motion is modified under viscous resistance, forming the baseline for fully coupled flow–particle systems.

---

## 🧠 How to Run

```bash
python main.py
