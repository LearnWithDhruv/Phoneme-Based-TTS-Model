import os
import csv
import librosa
import numpy as np
import torch
import warnings
from tqdm import tqdm

def prepare_dataset(metadata_path, audio_dir, phoneme_dir, sample_rate=22050, max_duration=15):
    dataset = []
    
    # Check if directories exist
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not os.path.exists(phoneme_dir):
        raise FileNotFoundError(f"Phoneme directory not found: {phoneme_dir}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Count total files first for better progress bar
    with open(metadata_path, 'r', encoding='utf-8') as f:
        total_files = sum(1 for _ in csv.DictReader(f))
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, total=total_files, desc="Processing audio files"):
            wav_file = row['wav_file']
            audio_path = os.path.join(audio_dir, wav_file)
            
            # Skip if audio file doesn't exist
            if not os.path.exists(audio_path):
                print(f"\nWarning: Audio file not found at {audio_path}, skipping...")
                continue
            
            # Get base filename without extension
            base_name = os.path.splitext(wav_file)[0]
            phoneme_file = os.path.join(phoneme_dir, f"{base_name}.txt")
            
            # Skip if phoneme file doesn't exist
            if not os.path.exists(phoneme_file):
                print(f"\nWarning: Phoneme file not found for {wav_file}, skipping...")
                continue
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Get duration
                    duration = librosa.get_duration(filename=audio_path)
                    if duration > max_duration:
                        print(f"\nWarning: {wav_file} is too long ({duration:.2f}s > {max_duration}s), skipping...")
                        continue
                    
                    # Load audio with resampling
                    audio, sr = librosa.load(
                        audio_path,
                        sr=sample_rate,
                        mono=True,
                        res_type='kaiser_fast'
                    )
                
                with open(phoneme_file, 'r', encoding='utf-8') as pf:
                    phonemes = pf.read().strip().split()
                
                # Convert to tensor
                audio_tensor = torch.tensor(audio, dtype=torch.float32)
                
                dataset.append({
                    'phonemes': phonemes,
                    'audio': audio_tensor,
                    'audio_path': audio_path
                })
                
            except Exception as e:
                print(f"\nError processing {wav_file}: {str(e)}")
                continue
    
    if not dataset:
        raise ValueError("No valid audio files were processed - dataset is empty")
    
    return dataset

if __name__ == "__main__":
    try:
        dataset = prepare_dataset(
            r'D:\phone\data\metadata.csv',
            r'D:\phone\data\audio_split',  # Use the split audio directory
            r'D:\phone\data\phonemes',
            max_duration=15
        )
        torch.save(dataset, r'D:\phone\data\dataset.pt')
        print(f"\nSuccessfully processed {len(dataset)} files")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")