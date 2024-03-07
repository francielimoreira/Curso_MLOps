import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import load_data
from model import MyModel
import matplotlib.pyplot as plt

# Função para treinar o modelo
def train(model, train_loader, test_loader, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    # Listas para visualização da perda e precisão
    loss_list = []
    accuracy_list = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 100 == 99:    # Imprime a cada 100 mini-lotes
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        
        # Avaliar o modelo no conjunto de testes ao final de cada época
        accuracy = evaluate(model, test_loader)
        
        # Armazenar perda e precisão para visualização
        loss_list.append(running_loss / len(train_loader))
        accuracy_list.append(accuracy)
        
        print("Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch+1, loss_list[-1], accuracy))
    
    # Plotar a curva de treinamento
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

# Função para avaliar o modelo
def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    # Carregar os dados
    train_loader, test_loader = load_data("corruptmnist")

    # Inicializar o modelo
    model = MyModel()

    # Treinar o modelo
    train(model, train_loader, test_loader)

