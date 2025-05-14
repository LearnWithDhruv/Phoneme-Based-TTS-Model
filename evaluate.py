import torch
import torch.nn.functional as F
import librosa
import numpy as np
import os
import csv

def kl_divergence(p, q, eps=1e-10):
    p = torch.tensor(p, dtype=torch.float32) + eps
    q = torch.tensor(q, dtype=torch.float32) + eps
    return F.kl_div(torch.log(p), q, reduction='batchmean').item()

def l1_loss(p, q):
    p = torch.tensor(p, dtype=torch.float32)
    q = torch.tensor(q, dtype=torch.float32)
    return torch.nn.L1Loss()(p, q).item()

def evaluate_audio(original_dir, synthesized_dir, metadata_path, sample_rate=22050):
    results = []
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        wav_files = [row['wav_file'] for row in reader][:3]
    
    for i, wav_file in enumerate(wav_files):
        orig_path = os.path.join(original_dir, wav_file)
        synth_path = os.path.join(synthesized_dir, f"synthesized_{i}.wav")
        
        if not os.path.exists(orig_path):
            print(f"Original audio not found: {orig_path}, skipping...")
            continue
        if not os.path.exists(synth_path):
            print(f"Synthesized audio not found: {synth_path}, skipping...")
            continue
        
        orig_audio, _ = librosa.load(orig_path, sr=sample_rate)
        synth_audio, _ = librosa.load(synth_path, sr=sample_rate)
        
        min_len = min(len(orig_audio), len(synth_audio))
        orig_audio = orig_audio[:min_len]
        synth_audio = synth_audio[:min_len]
        
        orig_mel = librosa.feature.melspectrogram(y=orig_audio, sr=sample_rate)
        synth_mel = librosa.feature.melspectrogram(y=synth_audio, sr=sample_rate)
        
        kl = kl_divergence(orig_mel, synth_mel)
        l1 = l1_loss(orig_audio, synth_audio)
        
        results.append({
            'example': i,
            'kl_divergence': kl,
            'l1_loss': l1
        })
        print(f"Example {i} ({wav_file}): KL Divergence = {kl:.4f}, L1 Loss = {l1:.4f}")
    
    return results

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_dir = os.path.join(base_dir, 'data', 'audio')
    synthesized_dir = os.path.join(base_dir, 'data', 'wav')
    metadata_path = os.path.join(base_dir, 'data', 'metadata.csv')
    evaluate_audio(original_dir, synthesized_dir, metadata_path)