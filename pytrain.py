import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(17)
np.random.seed(17)  

train_images = np.load('data/train_images.npy')   
train_labels = np.load('data/train_labels.npy')   
test_images  = np.load('data/test_images.npy')     
test_labels  = np.load('data/test_labels.npy')     

print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)




class MNISTNumpyDataset(Dataset):
    def __init__(self, images, labels):
        """
        images: numpy array of shape (N, 784)
        labels: numpy array of shape (N,)
        """
        # Convert NumPy arrays to PyTorch tensors.
        # We convert images to float (and you might want to normalize them) and labels to long (int64).
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_dataset = MNISTNumpyDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(SimpleMLP, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        Z = self.fc(x)  # Z: shape [B, output_dim]

        # Compute the row-wise maximum. 
        max_vals = torch.max(Z, dim=1, keepdim=True)[0]   # Shape: [B, 1]

        # Compute exponentials after subtracting max_vals (for numerical stability).
        exp_Z = torch.exp(Z - max_vals)                     # Shape: [B, output_dim]

        # Sum over the output dimension (each row).
        sumExp = torch.sum(exp_Z, dim=1, keepdim=True)        # Shape: [B, 1]
        
        # Normalize each row.
        A = exp_Z / sumExp 
        return A 


model = SimpleMLP(input_dim=784, output_dim=10)
learning_rate = 0.0001

W1 = model.fc.weight.detach().cpu().numpy()  # shape: [256, 784]
b1 = model.fc.bias.detach().cpu().numpy()      # shape: [256,]

W1_transposed = W1.T  

#np.savetxt("W1.txt", W1_transposed, fmt="%.6f")
#np.savetxt("b1.txt", b1, fmt="%.6f")

model.train()

eps = 1e-10  
nb = 0

for batch_idx, (images, labels) in enumerate(train_loader):

    #print(images[0])
    A = model(images)  
    log_probs = torch.log(A + eps)  # shape: [B, output_dim]
    selected_log_probs = log_probs.gather(dim=1, index=labels.view(-1, 1)).squeeze(1)

    loss = torch.mean(-selected_log_probs)
    
    
    
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    
    print(f"Batch {batch_idx+1}, Loss: {loss.item():.4f}")

    nb += 1
    if nb > 10 :
        break


# Alternatively, you can use nn.NLLLoss with reduction='sum':
# criterion = nn.NLLLoss(reduction='sum')
# loss = criterion(outputs, labels)
# Zero previous gradients (if any)