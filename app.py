"""
Interface web de demonstration du CycleGAN photo -> Monet.

Lancement :
    python app.py

L'interface s'ouvre sur http://127.0.0.1:7860
"""

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.generator import Generator

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Generateur photo -> Monet.
# Note sur la convention de nommage du projet : gen_A produit le domaine A
# (peintures Monet), gen_B produit le domaine B (photographies). Les fichiers
# fake_B sauvegardes pendant l'entrainement sont donc des sorties du generateur
# peinture -> photo.
CHECKPOINT = "checkpoints/gen_A.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Chargement du modele
# --------------------------------------------------------------------------
def load_generator(path: str) -> Generator:
    model = Generator(img_channels=3, num_features=64, num_residuals=9).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)

    # Le checkpoint peut etre soit un state_dict brut, soit un dictionnaire
    # contenant une cle "state_dict" (format utilise pendant l'entrainement).
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)

    model.eval()
    return model


generator = load_generator(CHECKPOINT)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def to_multiple_of_four(value: int) -> int:
    """Le generateur compresse l'image deux fois par 2, les dimensions
    doivent donc etre divisibles par 4."""
    return max(4, (value // 4) * 4)


@torch.inference_mode()
def stylize(image: Image.Image, resolution: int) -> Image.Image:
    if image is None:
        return None

    image = image.convert("RGB")

    # Redimensionnement en conservant le rapport d'aspect
    w, h = image.size
    scale = resolution / max(w, h)
    new_size = (to_multiple_of_four(int(w * scale)), to_multiple_of_four(int(h * scale)))
    image = image.resize(new_size, Image.LANCZOS)

    # Normalisation dans [-1, 1], identique a l'entrainement
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(DEVICE)

    output = generator(tensor)

    # Retour dans [0, 255]
    output = (output.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    output = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(output)


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
DESCRIPTION = """
Transfert de style non apparie realise avec un CycleGAN entraine de zero en PyTorch
sur le jeu de donnees monet2photo. Deposez une photo pour la voir reinterpretee
dans le style de Monet.

Les paysages, les scenes d'exterieur et les images comportant du ciel, de l'eau ou
de la vegetation donnent les meilleurs resultats : ce sont les sujets dominants du
corpus d'entrainement.
"""

with gr.Blocks(title="CycleGAN · Photo vers Monet") as demo:
    gr.Markdown("# Photo vers peinture de Monet")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        source = gr.Image(type="pil", label="Photo d'origine", sources=["upload", "clipboard"])
        result = gr.Image(type="pil", label="Interpretation Monet")

    resolution = gr.Slider(
        minimum=256,
        maximum=768,
        value=256,
        step=64,
        label="Resolution de traitement",
        info="256 correspond a la resolution d'entrainement. Au dela, le rendu est "
             "plus detaille mais la texture picturale s'affine et devient moins marquee.",
    )

    run = gr.Button("Transformer", variant="primary")

    run.click(fn=stylize, inputs=[source, resolution], outputs=result)
    source.upload(fn=stylize, inputs=[source, resolution], outputs=result)

    gr.Markdown(
        "Modele entraine environ 10 epoques sur une RTX 2060 Super (8 Go), "
        "en precision mixte, taille de lot 1, resolution 256 x 256."
    )

if __name__ == "__main__":
    demo.launch()
