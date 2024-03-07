import matplotlib.pyplot as plt
import torch

# Carregar os dados
corrupted_data = torch.load('/home/fran/Documentos/projeto_mlops/Pytorch/exercicio_final/corruptmnist/train_images_0.pt')

# Exibir algumas imagens
num_images_to_visualize = 5
fig, axes = plt.subplots(1, num_images_to_visualize, figsize=(10, 2))

for i in range(num_images_to_visualize):
    image = corrupted_data[i]  # Obter uma única imagem do tensor
    axes[i].imshow(image, cmap='gray')
    axes[i].axis('off')

plt.show()




