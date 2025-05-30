import torch
import gradio as gr
from model import GPT, GPTConfig
import tiktoken

# ===== 1. Load Model =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Same config you used for training
config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
model = GPT(config)
model.load_state_dict(torch.load("gpt124m_trained_final.pt", map_location=device))
model.to(device)
model.eval()

# ===== 2. Tokenizer =====
enc = tiktoken.get_encoding("gpt2")

def generate_reply(message, history=None):
    # Flatten prompt from history (if exists)
    prompt = ""
    if history:
        for user, bot in history:
            prompt += f"{user}\n{bot}\n"
    prompt += message

    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long)[None, :].to(device)

    with torch.no_grad():
        for _ in range(100):  # max 100 tokens
            logits, _ = model(input_ids)
            probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

    out = input_ids[0].tolist()
    return enc.decode(out[len(enc.encode(prompt)):])  # return only generated part

# ===== 3. Gradio Interface =====
chatbot = gr.ChatInterface(fn=generate_reply, title="Custom GPT Chatbot")
chatbot.launch()
