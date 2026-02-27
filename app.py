# app.py
# Professional Aesthetic AI Music Generator

import os
import torch
import gradio as gr
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wavfile

os.environ["HF_HOME"] = "./huggingface_cache"

MODEL_NAME = "facebook/musicgen-small"

print("Loading AI model... Please wait.")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()


def generate_music(mood, genre, description, duration):

    prompt = f"{mood} {genre} music. {description}"

    inputs = processor(text=[prompt], return_tensors="pt")

    max_tokens = int(duration) * 50

    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_tokens)

    sampling_rate = model.config.audio_encoder.sampling_rate
    audio = audio_values[0, 0].cpu().numpy()

    audio = audio / np.max(np.abs(audio))

    output_file = "generated_music.wav"
    wavfile.write(output_file, sampling_rate, audio.astype(np.float32))

    return output_file


# ✨ Custom Professional Styling
custom_css = """
body {
    background: linear-gradient(135deg, #141e30, #243b55);
}

.gradio-container {
    font-family: 'Segoe UI', sans-serif;
    max-width: 900px !important;
    margin: auto;
}

h1 {
    text-align: center;
    font-size: 2.8rem !important;
    font-weight: 700;
    letter-spacing: 1px;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.gr-box {
    border-radius: 15px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
}

button {
    background: linear-gradient(to right, #00c6ff, #0072ff) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

button:hover {
    transform: scale(1.03);
    transition: 0.2s ease-in-out;
}
"""


with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate"
    ),
    css=custom_css
) as demo:

    gr.Markdown("# 🎵 AI Music Remix & Mood Generator")
    gr.Markdown(
        "Generate studio-quality AI music based on mood and genre with intelligent remix generation."
    )

    with gr.Row():
        mood = gr.Dropdown(
            ["Happy", "Sad", "Energetic", "Calm", "Romantic", "Dark"],
            label="🎭 Mood"
        )

        genre = gr.Dropdown(
            ["Pop", "Classical", "EDM", "Jazz", "Rock", "Lo-fi"],
            label="🎼 Genre"
        )

    description = gr.Textbox(
        label="📝 Describe Your Music Vision",
        placeholder="Example: Cinematic emotional background score with soft piano and strings"
    )

    duration = gr.Slider(
        5, 15, value=8, label="⏳ Duration (seconds)"
    )

    generate_btn = gr.Button("🎧 Generate Music")

    output_audio = gr.Audio(
        label="🎶 Your Generated Track"
    )

    generate_btn.click(
        generate_music,
        inputs=[mood, genre, description, duration],
        outputs=output_audio
    )

demo.launch()