import os
import torch

# Caminho para o diretório contendo os arquivos de imagem
directory = '/home/fran/Documentos/projeto_mlops/Pytorch/exercicio_final/corruptmnist'

# Listar todos os arquivos no diretório
files = os.listdir(directory)

# Iterar sobre cada arquivo no diretório
for file in files:
    if file.endswith('.pt'):  # Verificar se é um arquivo .pt
        file_path = os.path.join(directory, file)  # Caminho completo do arquivo
        print("Arquivo:", file)
        
        # Carregar os dados do arquivo
        corrupted_data = torch.load(file_path)
        
        # Exibir o tipo de dados e o tamanho do tensor de imagens
        print("Tipo de dados:", type(corrupted_data))
        print("Tamanho do tensor de imagens:", corrupted_data.shape)
        print()  # Adicionar uma linha em branco entre cada arquivo











                        