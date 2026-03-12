# Picellium AI: Democratizing AI through Visual Novelty

![Picellium AI Logo](dev/nanochat.png)

## The Vision
I believe that we should democratize AI to the next level and provide a way for everyone to see AI work in visually novel ways. While AI is often seen as a "black box," Picellium AI aims to show AI working like organisms and insects—specifically leaf-cutter ants. I believe this organic, visual approach will make AI interesting and educational for all ages.

The plan is for many tiny AI agents to work together like a colony. Each agent has a very specific task and a short lifetime, but their memories are shared and instantly known by the entire colony.

## The Technology
Picellium AI is currently in its early stages, growing from the foundational work of Andrej Karpathy's **nanochat** (formerly nanoclaw).

- **Data Gathering:** I am using **OpenClaw** to coordinate and gather specialized data sets.
- **Specialized Models:** I am training small, task-specific models (Ants):
  - **Coding Supervisor Ant:** Trained on Uncle Bob's (Robert C. Martin) principles and practices.
  - **Rust Worker Ant:** Trained specifically in Rust coding to handle sections of code assigned by the supervisor.
- **Visual Experience:** The goal is a web-based platform where users ask questions and can visually watch the "Ant AI bots" go to work, just like a colony of leaf-cutter ants processing a leaf.

## About the Developer
I am a Senior Architect and web pioneer with a history of building high-impact systems:

- **AI & Horticulture:** Most recently, I was a Senior Architect at **Fargro**, where I built a team of AI and horticulture experts. We successfully helped **Thanet Earth** increase yields in controlled environments from 6% to 27%.
- **Enterprise Infrastructure:** I led the team that hosted the largest SharePoint implementation in Europe for the **Financial Ombudsman**.
- **Web & Music Industry Pioneer:** In the late 90s, I was the sole developer for **Sony Music Studios**, **Sony Music Classical**, **Wu Tang Clan**, **Reef**, **Fused**, and **Toploader**. During this era, I was known for pushing the boundaries of what was possible with pure JavaScript, proving that Flash extensions were not needed for rich experiences.

---

## Technical Foundation (nanochat)
Picellium AI is built on top of [nanochat](https://github.com/karpathy/nanochat), a minimal and hackable harness for training LLMs. It covers the entire lifecycle: tokenization, pretraining, finetuning, evaluation, and inference.

### Quick Start (Inherited from nanochat)
To run the base training or chat interface, ensure you have `uv` installed and follow these steps:

```bash
# Install dependencies
uv sync

# Talk to the model via CLI
python -m scripts.chat_cli -p "Hello!"

# Run the web UI
python -m scripts.chat_web
```

## License
MIT
