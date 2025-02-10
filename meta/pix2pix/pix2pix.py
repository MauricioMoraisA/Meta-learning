
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape, mean_absolute_error as mae, root_mean_squared_error as rmse
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
import sys 
import warnings
warnings.filterwarnings("ignore")

# Adicione o caminho do diretório que contém utlisCrosIn.py
import  sys
sys.path.append('./pix2pix')
from utils import *

loss_object = nn.BCEWithLogitsLoss()

# Função de perda para o Discriminador
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = loss_object(disc_generated_output, torch.zeros_like(disc_generated_output))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


# Função de perda para o Gerador
def generator_loss(disc_generated_output, gen_output, target, LAMBDA):
    gan_loss = loss_object(disc_generated_output, torch.ones_like(disc_generated_output))
    # Erro médio absoluto (L1 loss)
    l1_loss =  torch.mean(torch.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

# l1_loss = torch.mean(torch.abs(target - gen_output))
# mse_loss = torch.mean((target - gen_output) ** 2)
# total_gen_loss = gan_loss + (LAMBDA1 * l1_loss) + (LAMBDA2 * mse_loss)


def train_step(input_image, target_image, generator, discriminator, generator_optimizer, discriminator_optimizer):
    LAMBDA = 100
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    with torch.amp.autocast('cuda:0', enabled=True):  # mixed precision training
        # Gera a imagem
        generated_image = generator(input_image)
  
        # Calcula a perda do Discriminador
        disc_real_output = discriminator(target_image)
      
        disc_generated_output = discriminator(generated_image.detach())
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        
        # Calcula a perda do Gerador
        disc_generated_output = discriminator(generated_image)
        gen_loss, gan_loss, l1_loss = generator_loss(disc_generated_output, generated_image, target_image, LAMBDA)
    
    # Atualiza os pesos do Discriminador
    disc_loss.backward()
    discriminator_optimizer.step()
    
    # Atualiza os pesos do Gerador
    gen_loss.backward()
    generator_optimizer.step()
    
    return gen_loss, disc_loss, gan_loss, l1_loss


def train(epochs, train_loader, in_channels,taxa):
    # Inicialização de pesos
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    generator = Generator(img_channels=in_channels).to(DEVICE)
    discriminator = Discriminator(in_channels=in_channels).to(DEVICE)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Separar parâmetros
    norm_params = []
    other_params = []
    for name, param in generator.named_parameters():
        if 'norm' in name.lower() and param.requires_grad:
            norm_params.append(param)
        elif param.requires_grad:
            other_params.append(param)

    # Otimizador com taxas de aprendizado diferentes
    generator_optimizer = optim.Adam([
        {'params': other_params},
        {'params': norm_params, 'lr': 1e-5}
        ], lr=1e-4, betas=(0.5, 0.999))

    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # Criar schedulers para ajustar a taxa de aprendizado
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    gen_scheduler = ReduceLROnPlateau(generator_optimizer, mode='min', factor=0.5, patience=20, threshold=0.01, verbose=True)
    disc_scheduler = ReduceLROnPlateau(discriminator_optimizer, mode='min', factor=0.5, patience=20, threshold=0.01, verbose=True)

    early_stopping = EarlyStopping(patience=50, min_delta=0.001)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        total_gen_loss = 0
        total_disc_loss = 0

        for batch in train_loader:
            input_image, target_image, *rest = batch
            input_image, target_image = input_image.to(DEVICE), target_image.to(DEVICE)

            gen_loss, disc_loss, gan_loss, l1_loss = train_step(
                input_image, target_image, generator, discriminator,
                generator_optimizer, discriminator_optimizer
            )

            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()

            # Aplicar clipping de gradientes
            nn.utils.clip_grad_norm_(norm_params, max_norm=1.0)
            nn.utils.clip_grad_norm_(other_params, max_norm=5.0)

        avg_gen_loss = total_gen_loss / len(train_loader)
        avg_disc_loss = total_disc_loss / len(train_loader)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}')

        # Atualizar os schedulers
        gen_scheduler.step(avg_gen_loss)
        disc_scheduler.step(avg_disc_loss)

        # Checar o early stopping usando a perda de treinamento
        early_stopping(avg_gen_loss)

        if early_stopping.early_stop:
            print("Early stopping ativado.")
            break

        # Salvar o melhor modelo
        if early_stopping.counter == 0:
            torch.save(generator.state_dict(), f'{taxa}best_generator.pth')
            torch.save(discriminator.state_dict(), f'{taxa}best_discriminator.pth')

    # Carregar o melhor modelo antes de retornar
    generator.load_state_dict(torch.load(f'{taxa}best_generator.pth'))
    discriminator.load_state_dict(torch.load(f'{taxa}best_discriminator.pth'))

    return generator


def test(model,dataloader,dataset):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    imputed_data = {}
    for batch in dataloader:
        image_horse, image_zebra, mean, std, info = batch

        # Mover tensores para o dispositivo
        input_image = image_horse.to(device)

        with torch.no_grad():
            output = model(input_image)

        # Converter o tensor de saída para um array NumPy
        imputed_window = output.squeeze().cpu().numpy()

        # **Não desnormalizar aqui**

        # Reconstruir a série temporal a partir da janela
        imputed_series = imputed_window.flatten()

        # Armazenar os dados imputados para a coluna correspondente
        file_name = os.path.basename(info['file_input'][0])
        column = info['column'][0]

        if file_name not in imputed_data:
            imputed_data[file_name] = {}
        if column not in imputed_data[file_name]:
            imputed_data[file_name][column] = []

        imputed_data[file_name][column].append(imputed_series)

    # Após processar todas as janelas, reconstruir e salvar os dataframes
    for file_name in imputed_data:
        data = {}
        for column in imputed_data[file_name]:
            # Concatenar todas as janelas para formar a série completa
            series = np.concatenate(imputed_data[file_name][column])

            # Ajustar para o comprimento original, se necessário
            original_length = len(dataset.dataframes_horse[os.path.join(dataset.root_horse, file_name)][column])
            series = series[:original_length]

            data[column] = series

        # Reconstruir o dataframe
        dataset.reconstruct_dataframe(file_name, data)



TRAIN_DIR = '../imputation_Statistics/experiments/'
VAL_DIR = '../imputation_Statistics/experiments/'
BATCH_SIZE = 2048
NUM_EPOCHS = 2000

import time 

def main(in_channels):
    for taxa in [10, 20, 30]:
        print(f'treinando com Taxa {taxa}')
        
        dataset = Dataloadertimes(
            root_label=TRAIN_DIR + f"{taxa}/1024/treino/label",
            root_input=TRAIN_DIR + f"{taxa}/1024/treino/input",
            taxa=taxa,  # Add this line
            in_out='treino'
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        test_dataset = Dataloadertimes(
            root_label=VAL_DIR + f"{taxa}/1024/teste/label",
            root_input=VAL_DIR + f"{taxa}/1024/teste/input",
            taxa=taxa,
            in_out='teste'
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
        )

        # Treino
        start_time = time.time()
        generator = train(NUM_EPOCHS, loader, in_channels, taxa)
        end_time = time.time()

            # Calcula o tempo de execução
        execution_time = end_time - start_time
        print(taxa, execution_time)
        # Teste
        # Recreate the dataset and loader with taxa parameter
        dataset = Dataloadertimes(
            root_label=TRAIN_DIR + f"{taxa}/1024/treino/label",
            root_input=TRAIN_DIR + f"{taxa}/1024/treino/input",
            taxa=taxa,  # Add this line
            in_out='treino'
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

        # test(generator, test_loader, test_dataset)
        # test(generator, loader, dataset)

# import torch.multiprocessing as mp

if __name__ == '__main__':
    
    # mp.freeze_support()  # Necessário no Windows para compatibilidade
    for i in [1]:
        main(i)
