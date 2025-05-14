import torch
import soundfile as sf
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.nanospeech_torch import NanoSpeech
def synthesize_audio(model_path, phoneme_dir, output_dir, sample_rate=22050):
    os.makedirs(output_dir, exist_ok=True)
    
    model = NanoSpeech()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for i, phoneme_file in enumerate(sorted(os.listdir(phoneme_dir))[:3]):
        phoneme_path = os.path.join(phoneme_dir, phoneme_file)
        with open(phoneme_path, 'r', encoding='utf-8') as f:
            phonemes = f.read().strip().split()
        
        with torch.no_grad():
            audio = model(phonemes).cpu().numpy()
        
        output_path = os.path.join(output_dir, f"synthesized_{i}.wav")
        sf.write(output_path, audio, sample_rate)
        print(f"Synthesized: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'model', 'nanospeech', 'trained_model.pt')
    phoneme_dir = os.path.join(base_dir, 'data', 'phonemes')
    output_dir = os.path.join(base_dir, 'data', 'wav')
    synthesize_audio(model_path, phoneme_dir, output_dir)