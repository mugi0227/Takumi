# Takumi - Artisan's Skill Inheritance System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mugi007/takumi)

This system features Google's AI, Gemma 3n, acting as an expert "interviewer" to transform the un-manualizable "tacit knowledge" of skilled technicians into "explicit knowledge" that can be passed on to the next generation.

---

## 🚀 Live Demo

You can try the live application on Hugging Face Spaces:

**https://huggingface.co/spaces/mugi007/takumi**

---

## ✨ Core Features

This application is built on a unique **2-Phase Analysis Approach** to maximize the capabilities of Gemma 3n.

1.  **Phase 1: Auditory Analysis**
    * The AI first focuses solely on the audio from a work video. It transcribes not only speech but also critical environmental sounds (e.g., machine hums, tool clicks), creating a rich, timestamped auditory log.

2.  **Phase 2: Integrated Analysis & Dialogue**
    * The AI then analyzes the auditory log and video frames together. Instead of just summarizing, it acts as an **expert interviewer**, creating a draft manual (IKO Format) and generating insightful questions to probe the expert's tacit knowledge where information is missing.

3.  **Human-in-the-Loop**
    * The expert answers the AI's questions, and this new information is integrated to create a complete, high-quality technical manual.

---

## 🛠️ Tech Stack

* **Model:** Google Gemma 3n
* **Framework:** Gradio
* **Libraries:** Hugging Face Transformers, PyTorch, Accelerate, Bits and Bytes
* **Media Processing:** ffmpeg

---

## ⚙️ Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/takumi-skill-system.git](https://github.com/YOUR_USERNAME/takumi-skill-system.git)
    cd takumi-skill-system
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your Hugging Face Token:**
    * You will need a Hugging Face access token with access to Gemma models.
    * Create a file named `.env` in the root directory of the project.
    * Add your token to the `.env` file like this:
        ```
        HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
        ```

4.  **Run the application:**
    ```bash
    python app.py
    ```

---
