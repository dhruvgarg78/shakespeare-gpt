# 🧠 GPT-from-Scratch Chatbot

A lightweight GPT-style chatbot trained from scratch on Shakespeare’s *Coriolanus* using PyTorch. This project walks through building, training, and deploying a transformer-based language model with a Gradio interface.

## 🚀 Demo

Check out the live demo on Hugging Face Spaces:  
👉 [BardGPT – Shakespearean Chatbot](https://huggingface.co/spaces/dhruv78/GPT-124M-ScratchBot)

![image](https://github.com/user-attachments/assets/b31cfdcd-b5a1-4ae9-b29f-2af6b4c404c4)


## 🏗️ Architecture

- GPT-style decoder-only transformer
- 12 layers, 12 heads, 768 hidden units
- Trained from scratch on Shakespearean dialogue

## 📦 Files

| File         | Purpose                                     |
|--------------|---------------------------------------------|
| `model.py`   | Defines the GPT model (embedding, attention, layers) |
| `app.py`     | Gradio interface using the trained model    |
| `requirements.txt` | Dependencies for running the app      |

## 💬 How it works

After training the model to a loss of 0.098616 on Shakespeare’s *Coriolanus*, it is deployed as a chatbot using Gradio. It generates text in a Shakespearean style and mimics the format of a stage play.

![image](https://github.com/user-attachments/assets/00875945-5bad-4ee7-bd82-2667a5cfc001)

