# SVD-Based Kraus Operator Simulation

This repository contains a Jupyter notebook demonstrating the **unitary dilation of non-unitary evolution** using **Kraus operator decomposition** via **Singular Value Decomposition (SVD)**. The purpose is to simulate open quantum system dynamics — particularly under Lindblad-type noise — and implement the resulting dynamics using quantum circuits.

---

## 📘 Contents

- `SVD_Kraus.ipynb`: 
  - Constructs Kraus operators from a given Choi matrix
  - Performs SVD to unitarize non-unitary maps
  - Uses ancilla-based circuits to embed open system dynamics
  - Demonstrates implementation on a test state

---

## 🧠 Motivation

In noisy intermediate-scale quantum (NISQ) devices, decoherence and dissipation play a significant role. To simulate such open systems accurately, it's important to work with **non-unitary maps**, often modeled by the **Lindblad master equation**.

However, quantum circuits are unitary by construction. This notebook follows the method described in [Exact Non-Markovian Quantum Dynamics on the NISQ Device Using Kraus Operators (ACS Omega 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10906042/pdf/ao3c09720.pdf), where **Kraus operators are embedded into unitary operations using SVD** and **Walsh-based encoding**.

---

## 🛠️ Requirements

- Python 3.8+
- Qiskit
- NumPy
- SciPy
- Matplotlib (optional for visualization)
## Terminal commands 

Print Kraus operators, their SVD and the analytical rho(t) to terminal:

# Step 1: Simulate
python Lindblad_sim.py simulate --t 5.0 --gamma 1.0 --steps 50 --outdir ./outputs

# Step 2: Fidelity Plot
python Lindblad_sim.py fidelity --t 5.0 --gamma 1.0 --steps 50 --outdir ./outputs

# Step 3: Population Plot
python Lindblad_sim.py population --t 5.0 --gamma 1.0 --steps 50 --outdir ./outputs
