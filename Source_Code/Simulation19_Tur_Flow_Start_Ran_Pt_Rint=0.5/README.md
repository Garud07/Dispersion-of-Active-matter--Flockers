# 2d_active_matter
# Simulation19_Tur_Flow_Start_Ran_Pt_Rint=0.5

This repository contains the nineteenth stage of the *Dispersion of Active Matter in Turbulence* project.  
In this setup, a **turbulent flow field** is first evolved **without any particles** until it reaches a statistically steady state (~200 s).  
After the turbulence is fully developed, **particles are injected** into the flow with **random initial positions** and an **interaction radius `Rint = 0.5`**.

This approach allows the study of how pre-developed turbulence influences the initial dispersion and collective behavior of active particles.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** 2D forced turbulence (periodic domain)  
- **Flow Initialization:** Turbulence evolved alone for ~200 s (no particles)  
- **Particle Injection:** Random positions after turbulence becomes fully developed  
- **Dynamics (post-injection):**
  - **Self-propulsion:** Active velocity with noise  
  - **Stokes drag:** Linear viscous damping  
  - **Advection:** Coupling with the existing turbulent velocity field  
- **Interaction Radius (`Rint`):** 0.5  
- **Objective:**  
  To isolate the influence of a mature turbulent field on active particle clustering, dispersion, and alignment.

---

## 🧠 How to Run

```bash
python main.py
