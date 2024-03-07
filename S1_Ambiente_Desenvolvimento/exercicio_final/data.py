import os
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(data_dir):
    train_images = []
    train_targets = []
    test_images = []
    test_targets = []

    # Carregar dados de treinamento
    for i in range(6):
        train_images_file = os.path.join(data_dir, f"train_images_{i}.pt")
        train_targets_file = os.path.join(data_dir, f"train_target_{i}.pt")
        train_images.append(torch.load(train_images_file))
        train_targets.append(torch.load(train_targets_file))

    # Concatenar dados de treinamento
    train_images = torch.cat(train_images, dim=0)
    train_targets = torch.cat(train_targets, dim=0)

    # Carregar dados de teste
    test_images_file = os.path.join(data_dir, "test_images.pt")
    test_targets_file = os.path.join(data_dir, "test_target.pt")
    test_images = torch.load(test_images_file)
    test_targets = torch.load(test_targets_file)

    # Criar conjuntos de dados PyTorch
    train_dataset = TensorDataset(train_images, train_targets)
    test_dataset = TensorDataset(test_images, test_targets)

    # Criar carregadores de dados PyTorch
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

