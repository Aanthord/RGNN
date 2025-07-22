# 🧠 Recursive Graded Neural Networks (RGNN)

**RGNN** is a hybrid symbolic-recursive architecture combining PyTorch logic units with QASM-enhanced recursion. It is designed to model hierarchical symbolic computation, logic fidelity, and entropy-aware transformations.

This system bridges classical machine learning with symbolic reasoning and recursive quantum hooks, supporting NP-space tasks, SAT problems, and recursive learning depth.

---

## 🚀 Features

- 🧩 Recursive Unit (RU) modules for symbolic computation
- 🔄 Modular PyTorch model architecture
- 🔬 Entropy-aware contrastive loss
- 🔗 Qiskit-based QASM bridge for Φ / Φ′ logic gates
- 📊 Fidelity and entropy logging utilities
- 🧬 Supports symbolic tasks (SAT/NP, tree validation, etc.)
- ⚙️ Hooks for future RSSN + FTC operator injection


## 🛠️ Installation

```bash
git clone https://github.com/yourusername/rgnn_project.git
cd rgnn_project
pip install -r requirements.txt
```
Requires Python 3.9+, PyTorch 2.0+, Qiskit, NumPy, SciPy, Matplotlib

⸻

🧪 Quick Start

Train from scratch:

python main.py --train --config config.yaml

Run inference (CLI):

python main.py --infer --input "A OR NOT B -> C"

Evaluate a model:

python main.py --eval --checkpoint ./checkpoints/model.pt


⸻

📊 Visualize Fidelity

Run entropy and fidelity metrics:

python metrics.py --log ./logs/run1.json

Or inspect output patterns in notebooks/.

⸻

📚 Citation

If you use this project in academic research, please cite:

@misc{doran2025rgnn,
  title={Recursive Graded Neural Networks (RGNN)},
  author={Michael A. Doran Jr.},
  year={2025},
  note={https://github.com/aanthord/rgnn}
}


⸻

📜 License

This project is dual-licensed:

🔓 Academic & Personal Use (Free)

Permission is granted for academic research, teaching, and personal non-commercial experimentation free of charge, under the following conditions:
	•	You must cite the original author in any derivative or published work.
	•	You may not deploy, resell, or integrate this software into commercial applications or products.
	•	You may not train commercial AI models or services using this software without a commercial license.

⸻

💼 Commercial Use (Requires License)

Commercial use, including:
	•	SaaS or cloud deployment
	•	Product bundling or resale
	•	Use in enterprise AI/ML infrastructure
	•	Private training pipelines or tools

…requires a paid license and written agreement.

For licensing inquiries, contact:
	•	📩 michael.doran.808@gmail.com
	•	🔗 linkedin.com/in/michaeldoranjr

⸻

🧬 Author

Michael A. Doran Jr.
Polymath, systems architect, and builder of recursive symbolic infrastructure.

⸻

🛤️ Future Work
	•	🧠 RSSN operator integration
	•	🧮 Fractal Tensor Calculus backend
	•	🔐 MerkleCube verification on inference traces
	•	📈 TorchScript export and model cards

