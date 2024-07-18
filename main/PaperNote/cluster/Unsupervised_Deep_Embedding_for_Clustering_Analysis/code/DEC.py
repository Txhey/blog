import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torchvision import datasets, transforms


# Step 1: Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)


# Step 2: Pretrain the Autoencoder
def pretrain_autoencoder(autoencoder, data_loader, num_epochs=20, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for data, _ in data_loader:
            data = data.view(data.size(0), -1)
            output = autoencoder(data)
            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Step 3: Initialize Clustering Centers using KMeans
def initialize_cluster_centers(encoder, data_loader, n_clusters):
    features = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1)
            encoded_data = encoder(data)
            features.append(encoded_data)
    features = torch.cat(features).numpy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(features)
    cluster_centers = kmeans.cluster_centers_
    return torch.tensor(cluster_centers, dtype=torch.float32), y_pred


# Step 4: Define the DEC Model
class DEC(nn.Module):
    def __init__(self, autoencoder, n_clusters):
        super(DEC, self).__init__()
        self.autoencoder = autoencoder
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, autoencoder.encoder[-1].out_features))

    def forward(self, x):
        return self.autoencoder.encode(x)

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T


# Step 5: Train the DEC Model
def train_DEC(dec_model, data_loader, y_pred, num_epochs=100, learning_rate=1e-3, update_interval=10):
    optimizer = optim.Adam(dec_model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        if epoch % update_interval == 0:
            features = []
            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.view(data.size(0), -1)
                    encoded_data = dec_model(data)
                    features.append(encoded_data)
            features = torch.cat(features).numpy()
            q = 1.0 / (1.0 + pairwise_distances_argmin_min(features, dec_model.cluster_centers.cpu().numpy())[1])
            q = (q.T / q.sum(1)).T
            p = dec_model.target_distribution(torch.tensor(q))

        for data, _ in data_loader:
            data = data.view(data.size(0), -1)
            encoded_data = dec_model(data)
            q = 1.0 / (1.0 + pairwise_distances_argmin_min(encoded_data.cpu().detach().numpy(),
                                                           dec_model.cluster_centers.cpu().numpy())[1])
            q = (q.T / q.sum(1)).T
            kl_loss = nn.KLDivLoss(reduction='batchmean')(torch.log(torch.tensor(q)), p)
            optimizer.zero_grad()
            kl_loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {kl_loss.item():.4f}')


# Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

input_dim = 28 * 28
hidden_dim = 10
n_clusters = 10

# Instantiate Autoencoder and DEC model
autoencoder = Autoencoder(input_dim, hidden_dim)
dec_model = DEC(autoencoder, n_clusters)

# Pretrain the Autoencoder
pretrain_autoencoder(autoencoder, train_loader)

# Initialize Clustering Centers
cluster_centers, y_pred = initialize_cluster_centers(autoencoder.encode, train_loader, n_clusters)
dec_model.cluster_centers.data = cluster_centers

# Train the DEC Model
train_DEC(dec_model, train_loader, y_pred)
