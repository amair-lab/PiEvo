# Principle-Evolvable Scientific Discovery via Uncertainty Minimization

<div align="center">

[Yingming Pu](dandelionym.github.io) &emsp; [Tao Lin](https://lins-lab.github.io/) &emsp; [Hongyu Chen](https://nanosynthesis.github.io/)

Westlake University

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)&emsp;
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)&emsp;
[![AutoGen](https://img.shields.io/badge/AutoGen-0.7.4-green)](https://github.com/microsoft/autogen)&emsp;
[![ArXiv](https://img.shields.io/badge/arXiv-2602.06448-red)](https://arxiv.org/abs/2602.06448)

[Features](#key-features) • [Installation](#installation) • [Quick Start](#quick-start) • [Citation](#citation)

</div>

---

## 📖 Abstract

Large Language Model (LLM)-based scientific agents have accelerated scientific discovery, yet they often suffer from significant inefficiencies due to adherence to fixed initial priors. Existing approaches predominantly operate within a static hypothesis space, which restricts the discovery of novel phenomena, resulting in computational waste when baseline theories fail. To address this, we propose shifting the focus from searching hypotheses to evolving the underlying scientific principles. We present PiEvo, a principle-evolvable framework that treats scientific discovery as Bayesian optimization over an expanding principle space. By integrating Information-Directed Hypothesis Selection via Gaussian Process and an anomaly-driven augmentation mechanism, PiEvo enables agents to autonomously refine their theoretical worldview. Evaluation across four benchmarks demonstrates that PiEvo (1) achieves an average solution quality of up to 90.81%~93.15%, representing a 29.7%~31.1% improvement over the state-of-the-art, (2) attains an 83.3% speedup in convergence step via significantly reduced sample complexity by optimizing the compact principle space, and (3) maintains robust performance across diverse scientific domains and LLM backbones.

<div align="center">
  <img src="assets/illustration.png" alt="PiEvo Illustration" width="400"/>
</div>


---

## ✨ Key Features

- **🤖 Role-Based Collaboration**: Specialized agents (Principle, Hypothesis, Experiment) work in a closed feedback loop.
- **🧠 Knowledge Evolution**: Continuously refines its scientific understanding, storing successful principles in a long-term memory.
- **🛡️ Secure Execution**: All code generation and execution are sandboxed within Docker containers to ensure safety and reproducibility.
- **⚡ High Performance**: Built on top of Microsoft's AutoGen and optimized for parallel execution.
- **📊 Comprehensive Monitoring**: Tracks every step of the discovery process, from initial hypothesis to final experimental validation.



---

## 🛠️ Installation

Prerequisites:
- Python 3.12 or higher
- PyTorch
- LLM API key (e.g., OpenAI API key)

clone the repository and install the package in editable mode:

```bash
git clone https://github.com/your-org/PiEvo.git && cd PiEvo

# We recommand to install PiEvo in a virtual environment like conda:
conda create -n pievo python=3.12
conda activate pievo
pip install -e .
```

### Dependencies
PiEvo relies on several key libraries including `autogen-agentchat==0.7.4`, `openai>=1.52`, `numpy==1.23`, and `scipy`. A full list can be found in `pyproject.toml`. 

---

## 🚀 Quick Start

To run a demo of the PiEvo system, use the provided configuration files in the `config` directory. You should first set the LLM api info in `config/model.yaml`. Then, the following command initiates a scientific discovery task:

```bash
python pievo/main.py \
    --task_config config/task.yaml \
    --model_config config/model.yaml \
    --output_dir ./outputs \
    --max_turn 20
```

Note that, this demo of Nanohelix Optimization (NHO) will produce random result as a quick example.   For any realworld task, you can modify and config the tools in `tools` directory, for more details, please refer to [PiFlow](https://github.com/amair-lab/PiFlow).


### Arguments
- `--task_config`: Path to the YAML file defining the scientific task.
- `--model_config`: Path to the YAML file configuring the LLM agents.
- `--output_dir`: Directory where results and logs will be saved.
- `--max_turn`: Maximum number of conversation turns for the session.

---

## 📄 Citation

If you use PiEvo in your research, please cite our papers:

```bibtex
@misc{pu2026pievo,
      title={Principle-Evolvable Scientific Discovery via Uncertainty Minimization}, 
      author={Yingming Pu and Tao Lin and Hongyu Chen},
      year={2026},
      eprint={2602.06448},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.06448}, 
}

@misc{pu2025piflow,
      title={PiFlow: Principle-Aware Scientific Discovery with Multi-Agent Collaboration}, 
      author={Yingming Pu and Tao Lin and Hongyu Chen},
      year={2026},
      eprint={2505.15047},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.15047}, 
}
```

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.