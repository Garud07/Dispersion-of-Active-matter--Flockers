# 2d_active_matter
# Simulation7_RanUni_Stokesian_only

This repository contains the seventh stage of the active matter–turbulence project.  
In this case, particles are **randomly and uniformly initialized** in a periodic 2D domain, but evolve **purely under Stokesian dynamics** — i.e. **viscous drag and thermal (Brownian) forcing**, with **no self-propulsion component**.

This serves as the control case to compare passive particle dispersion with active/self-propelled systems.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** None  
- **Initial Distribution:** Random and spatially uniform  
- **Dynamics:** Stokesian motion with viscous drag and Brownian forcing  
- **Self-Propulsion:** Disabled  
- **Objective:**  
  Establish a baseline for passive particle dispersion in viscous media, serving as a comparison for self-propelled and coupled (active–flow) cases.

---

## 🧠 How to Run

```bash
python main.py
