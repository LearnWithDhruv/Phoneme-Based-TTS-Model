import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.nanospeech_torch import NanoSpeech

def train_model(dataset_path, epochs=10, batch_size=16, sample_rate=22050):
    dataset = torch.load(dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = NanoSpeech() 
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            phonemes = batch['phonemes']
            audio = batch['audio'].to(device)
            
            optimizer.zero_grad()
            predicted_audio = model(phonemes)
            loss = criterion(predicted_audio, audio)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model', 'nanospeech')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'trained_model.pt'))
    
    # Plot loss
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(plots_dir, 'loss_plot.png'))
    plt.close()

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.pt')
    train_model(dataset_path)