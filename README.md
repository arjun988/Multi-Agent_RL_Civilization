# Multi-Agent RL Civilization
### Research-grade multi-agent RL civilization simulation project.

# RL Architecture & Concepts: Organism Living System 

This version represents a significant leap from previous versions by introducing **Hierarchical RL**, **World Models**, and **Transformer-based memory**, moving beyond standard MAPPO with basic GRU architectures.

---

## 🚀 1. RL Concepts Used

### 1.1 Hierarchical Reinforcement Learning (HRL)
V6 breaks down complex behaviors into a multi-layered hierarchy.
* **Definition:** A framework that uses a **High-Level Policy (Manager)** to set abstract goals and a **Low-Level Policy (Worker)** to execute primitive actions to reach those goals.
* **Mathematical Explanation:**
    * The high-level policy $\pi_{high}(g|s)$ samples a goal $g$.
    * The low-level policy $\pi_{low}(a|s, g)$ is conditioned on that goal.
    * **Reward:** $R = r_{extrinsic} + r_{intrinsic}(s, g)$, where the low-level is rewarded specifically for goal-reaching behaviors.



### 1.2 World Model (Dreamer/Hallucination)
Inspired by the Dreamer architecture, agents "hallucinate" future states to learn without constant environmental interaction.
* **Definition:** A model that learns to predict the next state $s_{t+1}$ and reward $r_t$ given current state $s_t$ and action $a_t$.
* **Mathematical Explanation:**
    * **Transition Model:** $P(s_{t+1} | s_t, a_t)$
    * **Imagination Training:** Agents update their policies by acting within this learned model (hallucination) to simulate consequences before they happen in the real world.

### 1.3 Multi-Agent PPO (MAPPO) with CTDE
V6 utilizes MAPPO with an upgraded **Centralized Training with Decentralized Execution** (CTDE) pipeline.
* **Definition:** The **Critic** has access to global information (all agent positions, resource maps), while the **Actor** only sees the "Fog of War" local observation.
* **Objective Function:**
$$L_{CLIP}(\theta) = \mathbb{E}_t [ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+ \epsilon) A_t) ]$$



### 1.4 Transformer Memory (8-Step Attention)
To handle long-range dependencies better than standard GRUs, v6 introduces spatial-temporal attention.
* **Definition:** Uses an attention mechanism to weigh the importance of past observations over a fixed window.
* **Mathematical Explanation:**
$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
* In V6, $Q, K, V$ are derived from the **8 most recent time-steps**, allowing the agent to "attend" to specific past events (like the last known location of a predator).

---

## 🏗️ 2. Architecture

### 🔷 High-Level Flow
1. **Environment:** 2D World with Fog of War and multi-resource nodes.
2. **Transformer Window:** Last 8 observations are processed via Self-Attention.
3. **HRL Controller:** High-level sets a "Goal Embedding" (e.g., "Find Food" or "Defend Village").
4. **Actor-Critic Core:**
    * **Actor:** Combines Attention Context + Goal Embedding + GRU state $\to$ Action Logits.
    * **Critic:** Processes Global State $\to$ Value $V(s)$.
5. **Ensemble Output:** MAPPO output is merged with **NEAT** (NeuroEvolution of Augmenting Topologies) output for final action selection.

### 🧠 Neural Network Design: "Integrated Civ-Brain"
* **Encoder:** Linear layers for local spatial and status data.
* **Memory Layer 1 (Transformer):** 8-step temporal attention window.
* **Memory Layer 2 (GRU):** Recurrent state for long-term persistence beyond the attention window.
* **Action Heads:** * **Primary:** (Move, Eat, Attack, Trade, Communicate).
    * **Communication:** 8-dim learned message vectors for social coordination.

---

## 📊 3. Training Pipeline

* **Self-Play Archive:** Agents train against current and historical versions of themselves to ensure robust evolution.
* **PBT (Population Based Training):** Hyperparameters (like curiosity weights) are evolved in real-time based on population fitness.
* **Auto-Curriculum:** Environment difficulty (resource scarcity, disaster frequency) scales dynamically based on the current population's average health.

---

## ⚙️ 4. Key Hyperparameters (v6)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate** | `1.2e-4` | Optimized for Transformer stability |
| **Gamma ($\gamma$)** | `0.99` | Reward discount factor |
| **GAE Lambda ($\lambda$)** | `0.95` | Advantage smoothing |
| **Entropy Coef** | `0.018` | Lowered from v5 to encourage policy specialization |
| **Curiosity Coef** | `0.025` | Intrinsic exploration reward scaling |
| **Attention Window** | `8` | Number of past steps the Transformer "sees" |
| **Batch Size** | `1024` | Transitions sampled per gradient update |

---

## Repository Structure

```text
Multi-Agent_RL_Civilization/
├─ configs/
├─ docs/
│  ├─ architecture/
│  │  └─ V6 architechture.pdf
│  └─ research/
│     └─ CivilizationV7_README.docx
├─ notebooks/
│  └─ RL_Civilization.ipynb
├─ outputs/
│  └─ figures/
│     ├─ HRL.png
│     └─ MAPPO.png
├─ scripts/
│  └─ run_simulation.py      # Script entrypoint
├─ src/
│  └─ civilization/
│     ├─ __init__.py
│     ├─ __main__.py         # Enables: python -m civilization
│     ├─ simulation.py       # Canonical simulation implementation
│     ├─ core/               # Config, enums, entities
│     ├─ rl/                 # Models, policies
│     ├─ env/                # World/environment APIs
│     ├─ viz/                # Renderer APIs
│     └─ train/              # Runner APIs
├─ .gitignore
├─ LICENSE
├─ pyproject.toml
├─ requirements.txt
└─ README.md


## Setup Environment (Local)

### Option A: Virtual Environment + `requirements.txt` (recommended)

```powershell
# from repo root
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Editable Install (`pyproject.toml`)

```powershell
# from repo root
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
```

---

## Run the Code (Local)

From the project root, use any one of the following:

```powershell
# 1) Run as package module
python -m civilization

# 2) Run the script entrypoint
python scripts/run_simulation.py

# 3) If installed with `pip install -e .`
civ-sim
```

---

## Run Directly Without Local Setup (Google Colab)

You can run the notebook directly in Colab without creating a local environment.

1. Download `notebooks/RL_Civilization.ipynb`.
2. Open [Google Colab](https://colab.research.google.com/).
3. Upload the notebook (`File -> Upload notebook`).
4. Set runtime to GPU (`Runtime -> Change runtime type -> GPU`).
5. Run all cells.

The notebook already includes dependency installation cells, so this path works even if you do not set up Python locally.

