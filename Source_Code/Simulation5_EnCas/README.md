# 2d_active_matter
# Simulation5_EnCas

This simulation contains the case 
of the active where we focus on matter–turbulence.  
In this case, **energy is injected at 
different wavenumber bands** to investigate 
the **energy cascade** mechanism and its 
influence on particle dispersion and collective dynamics.

---

## 📁 Project Structure

---

## ⚙️ Simulation Details

- **Flow Type:** 2D turbulent flow with **energy injection at non-standard wavenumbers**  
- **Method:** Spectral formulation (Fourier basis, Dedalus-style)  
- **Forcing:** Band-limited stochastic forcing to sustain turbulence at desired scales  
- **Timestepping:** RK222 / RK4 (semi-implicit)  
- **Particles:** Passive/active agents advected within the flow  
- **Objective:** Examine how different energy injection scales influence particle motion, clustering, and alignment.

---

## 🧠 How to Run

```bash
python main.py



