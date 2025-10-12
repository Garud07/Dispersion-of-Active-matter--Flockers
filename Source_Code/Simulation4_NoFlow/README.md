# 2d_active_matter
# Simulation4_NoFlow

This repository contains the fourth 
stage of my active matter study, 
focusing on particle dynamics 
**without any background flow field**.  
It serves as the baseline test for 
self-propelled (Vicsek-type) or 
Brownian particles, allowing direct 
comparison with turbulent or vortical 
flow cases.

---

## 📁 Project Structure

---

## ⚙️ Simulation Details

- **Flow Field:** None (U = 0)  
- **Domain:** 2D periodic square box  
- **Particle Type:** Active / Vicsek-style self-propelled agents  
- **Forces:** Alignment + stochastic noise  
- **Integration:** Explicit time-stepping (Euler / RK2)  
- **Outputs:**
  - Particle trajectories
  - Order parameter and time series
  - Animations and statistical plots

---

## 🧠 How to Run

```bash
python main.py


