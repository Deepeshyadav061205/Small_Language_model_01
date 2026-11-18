# Small_Language_model_01

# 📖 TinyStories SLM: A Topic-Conditioned Small Language Model

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-TinyStories-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/roneneldan/TinyStories)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepeshyadav061205/Small_Language_model_01/blob/main/small_language_model.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green)](#-license)

> A small GPT-style language model, **built from scratch**, that writes short children's stories from a topic such as `dog, park, ball`.

---

## 🧭 Quick Navigation

- [✨ Features](#-features)
- [🎬 Demo](#-demo)
- [🏗️ Architecture](#️-architecture)
- [📚 Dataset](#-dataset)
- [⚙️ Training](#️-training)
- [📈 Results](#-results)
- [🚀 Getting Started](#-getting-started)
- [⚠️ Limitations](#️-limitations)
- [👥 Team](#-team)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **From-scratch model** | A decoder-only Transformer (GPT-style) written in PyTorch, with no pretrained weights |
| 🔤 **Custom tokenizer** | Byte-level BPE trained on the dataset (8,000-token vocabulary) |
| 🎯 **Topic conditioning** | Give keywords and get a story that uses them |
| 💾 **Reusable weights** | Trained model and tokenizer are included, so no retraining is needed |
| 📊 **Loss tracking** | Training and validation curves are saved and plotted |
| 🌐 **Web interface** | A Gradio app with a text box, sliders for length, creativity, and top-k, and example prompts |

---

## 🎬 Demo

Sample prompt format:

```text
Topic: dog, park, ball
Story:
```

<details>
<summary><b>👉 Click to see a sample output (500 training steps)</b></summary>

<br>

```text
TOPIC: dog, park, ball
Lily went to play together. Lily. Lily loved to see that and Tom asked a big...
```

> Note: at this training stage the model produces story-like text, but it does not reliably use every topic word. See [Limitations](#️-limitations).

</details>

<details>
<summary><b>🌐 Run the web app</b></summary>

<br>

1. Open the notebook with the **Open In Colab** badge above and set the runtime to a GPU if available.
2. Run all cells in order. The last cell (the Gradio cell) clones this repository to load the trained model.
3. Open the public link it prints, ending in `gradio.live`.
4. Enter topic words such as `dragon, cave, gold`, adjust the sliders, and click **Submit**.

**Slider guide:**
- **Story length:** how many tokens to generate.
- **Creativity (temperature):** lower gives safer, more repetitive text; higher gives more varied text.
- **Top-k sampling:** limits the model to its k most likely next tokens at each step.

The link stays active while the Colab session runs.

</details>

<details>
<summary><b>🧪 Generate a story in code</b></summary>

<br>

```python
story_about("dragon, cave, gold")
```

</details>

---

## 🏗️ Architecture

<details>
<summary><b>Click to expand the model architecture</b></summary>

<br>

```
Input tokens ──► Token Embedding + Position Embedding
                        │
                        ▼
        ┌─────────────────────────────────┐
        │  Transformer Block  (× N)       │
        │  ├─ LayerNorm                   │
        │  ├─ Causal Multi-Head Attention │
        │  ├─ Residual connection         │
        │  ├─ LayerNorm                   │
        │  ├─ MLP (GELU, 4× expansion)    │
        │  └─ Residual connection         │
        └─────────────────────────────────┘
                        │
                        ▼
                  Final LayerNorm
                        │
                        ▼
          Linear head (weight-tied to embedding)
                        │
                        ▼
              Next-token probabilities
```

</details>

| Hyperparameter | Value |
|---|---|
| Parameters | **1.45M** |
| Layers (`n_layer`) | 2 |
| Attention heads (`n_head`) | 2 |
| Embedding size (`n_embd`) | 128 |
| Context length (`block_size`) | 256 |
| Vocabulary size | 8,000 (BPE) |
| Dropout | 0.1 |
| Weight tying | ✅ Yes |

---

## 📚 Dataset

- **Source:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Ronen Eldan and Yuanzhi Li
- **Used:** 20,000 randomly sampled stories (seed 42)
- **Topic extraction:** each story gets 3 keywords taken from the story itself. Stop-words are removed, and only words that appear in 20–5,000 stories are kept.
- **Training format:**

```text
Topic: dog, park, ball
Story: Tim and Mia like to play in the park. They see a big ball...
```

---

## ⚙️ Training

<details>
<summary><b>Click to see the training configuration</b></summary>

<br>

| Setting | Value |
|---|---|
| Optimizer | AdamW (weight decay 0.1) |
| Learning rate | Warmup to `1e-3`, then cosine decay to `1e-4` |
| Batch size | 16 |
| Training steps | 500 |
| Gradient clipping | 1.0 |
| Hardware | CPU (Google Colab) |
| Training time | ~10 minutes |

</details>

---

## 📈 Results

| Step | Train loss | Validation loss |
|:---:|:---:|:---:|
| 0 | 9.00 | 9.00 |
| 250 | 4.04 | 3.98 |
| 499 | **3.77** | **3.79** |

- Starting loss of about 9.0 matches a random model, since `ln(8000) ≈ 9.0`.
- The loss curve is saved as [`model/loss_curve.png`](model/loss_curve.png).

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Deepeshyadav061205/Small_Language_model_01.git
cd Small_Language_model_01
```

### 2️⃣ Install dependencies

```bash
pip install torch tokenizers datasets matplotlib gradio
```

### 3️⃣ Load the trained model

The model class (`GPT`) is defined in the notebook, so run the notebook's model cell first, then load the saved weights:

```python
import torch
from tokenizers import ByteLevelBPETokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = ByteLevelBPETokenizer(
    "model/tinystories_tokenizer/vocab.json",
    "model/tinystories_tokenizer/merges.txt",
)
eos_id = tokenizer.token_to_id("<|endoftext|>")

checkpoint = torch.load("model/slm_topic_stories.pt", map_location=device)
model = GPT(**checkpoint["config"]).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

---

## ⚠️ Limitations

<details>
<summary><b>Click to read the known limitations</b></summary>

<br>

- **Undertrained:** 500 steps on CPU with a small model. Loss was still falling, so longer training would help.
- **Topic adherence is weak:** the model doesn't always include the requested keywords.
- **Vocabulary coverage:** rare words like `xyz` or `spaceship` are poorly learned, because TinyStories uses simple vocabulary.
- **No instruction following:** it responds to topic keywords only, not to full sentences like "write a story about a robot."

**Planned improvements:**
- [ ] Train for 3,000+ steps on a GPU
- [ ] Use the larger configuration (`n_layer=6`, `n_head=6`, `n_embd=384`)
- [ ] Compare against the plain-continuation baseline
- [ ] Host the Gradio app permanently on Hugging Face Spaces

</details>

---

## 👥 Team

| Role | Name | GitHub |
|---|---|---|
| 🏆 Team Leader | Deepesh Yadav | [@Deepeshyadav061205](https://github.com/Deepeshyadav061205) |
| 👤 Member | Ravikant | [@handle](https://github.com/) |
| 👤 Member | Chavi Prakash | [@handle](https://github.com/) |
| 👤 Member | Deepak Patel | [@handle](https://github.com/) |

> 🎓 *Project completed under the guidance of **Dhruv Sir**.*

---

## 📄 License

This project is released under the MIT License.

---

<p align="center">
  ⭐ If you found this project useful, consider giving it a star! ⭐
</p>
