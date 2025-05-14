import csv
import os
from gruut import sentences

def phonemize_texts(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcript = row['transcript']
            wav_file = row['wav_file']
            
            phonemes = []
            for sent in sentences(transcript, lang='en-us'):
                for word in sent:
                    if hasattr(word, 'phonemes') and word.phonemes:
                        phonemes.extend(word.phonemes)
            
            phoneme_str = ' '.join(phonemes)
            
            phoneme_file = os.path.join(output_dir, f"{os.path.splitext(wav_file)[0]}.txt")
            with open(phoneme_file, 'w', encoding='utf-8') as pf:
                pf.write(phoneme_str)

if __name__ == "__main__":
    phonemize_texts(
        r'D:\phone\data\metadata.csv', 
        r'D:\phone\data\phonemes'
    )