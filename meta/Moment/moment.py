
from momentfm.utils.utils import control_randomness
control_randomness(seed=13) # Set random seeds for PyTorch, Numpy etc.

from momentfm import MOMENTPipeline

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly")


class ImputationDataset(Dataset):
    def __init__(
        self,
        input_path: str,
        label_path: str,
        seq_len: int = 512,
        stride: int = 512  # Stride igual ao tamanho da janela para janelas não sobrepostas
    ):
        """
        Args:
            input_path (str): Caminho para os arquivos de entrada (com valores faltantes).
            label_path (str): Caminho para os arquivos de rótulo (dados completos).
            seq_len (int): Tamanho da janela de sequência.
            stride (int): Passo entre as sequências (igual a seq_len para janelas não sobrepostas).
        """
        self.seq_len = seq_len
        self.stride = stride
        self.input_path = input_path
        self.label_path = label_path

        # Inicializar o scaler
        self.scaler = StandardScaler()

        # Processar os dados e preparar as sequências
        self.process_data()

    def process_data(self):
        """
        Processa os dados:
        1. Lê os arquivos CSV.
        2. Corta as linhas que não se encaixam no tamanho da janela.
        3. Achata os dados para uma única coluna se houver múltiplas.
        4. Normaliza os dados.
        5. Concatena todas as séries em um único vetor.
        6. Gera máscaras para valores ausentes.
        7. Cria as sequências de treinamento.
        """
        all_input_data = []
        all_label_data = []

        input_files = sorted(os.listdir(self.input_path))
        label_files = sorted(os.listdir(self.label_path))

        # Garantir que ambos os diretórios tenham o mesmo número de arquivos
        assert len(input_files) == len(label_files), "Número diferente de arquivos em input e label."

        for input_file, label_file in zip(input_files, label_files):
            input_file_path = os.path.join(self.input_path, input_file)
            label_file_path = os.path.join(self.label_path, label_file)

            # Ler os arquivos CSV
            df_input = pd.read_csv(input_file_path, index_col=0)
            df_label = pd.read_csv(label_file_path, index_col=0)

            # Garantir que ambos os DataFrames tenham as mesmas colunas
            common_columns = df_input.columns.intersection(df_label.columns)
            df_input = df_input[common_columns].copy()
            df_label = df_label[common_columns].copy()

            # Achatar os dados para uma única coluna (concatenar todas as séries)
            input_series = df_input.values.flatten()
            label_series = df_label.values.flatten()

            all_input_data.append(input_series)
            all_label_data.append(label_series)

        # Concatenar todas as séries em um único vetor
        concatenated_input = np.concatenate(all_input_data)
        concatenated_label = np.concatenate(all_label_data)

        # Determinar o número de janelas possíveis
        total_length = len(concatenated_input)
        num_sequences = (total_length - self.seq_len) // self.stride + 1

        # Ajustar o tamanho dos vetores para que caibam nas janelas
        adjusted_length = self.stride * num_sequences + self.seq_len
        concatenated_input = concatenated_input[:adjusted_length]
        concatenated_label = concatenated_label[:adjusted_length]

        # Reshape para 2D (n_samples, 1) para o scaler
        concatenated_input = concatenated_input.reshape(-1, 1)
        concatenated_label = concatenated_label.reshape(-1, 1)

        # Normalizar os dados usando o scaler ajustado nos dados de label
        self.scaler.fit(concatenated_label)  # Ajustar apenas nos dados de label para evitar vazamento de informação
        concatenated_label = self.scaler.transform(concatenated_label)
        concatenated_input = self.scaler.transform(concatenated_input)

        # Reshape de volta para 1D
        concatenated_input = concatenated_input.flatten()
        concatenated_label = concatenated_label.flatten()

        # Gerar máscaras (1 onde há dados, 0 onde há NaN)
        # Supondo que os valores ausentes são representados como NaN nos dados de entrada
        mask = (~np.isnan(concatenated_input)).astype(float)  # 1 para dados presentes, 0 para ausentes

        # Preencher NaNs com zero nos dados de entrada
        mask = (~np.isnan(concatenated_input)).astype(float)  # 1 para dados presentes, 0 para ausentes
        concatenated_input = np.nan_to_num(concatenated_input, nan=np.nanmean(concatenated_input))#

        # Criar as sequências
        self.sequences = []
        for i in range(num_sequences):
            start = i * self.stride
            end = start + self.seq_len

            input_seq = concatenated_input[start:end]
            label_seq = concatenated_label[start:end]
            mask_seq = mask[start:end]

            # Converter para tensores
            input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)   # [1, seq_len]
            label_seq = torch.tensor(label_seq, dtype=torch.float32).unsqueeze(0)   # [1, seq_len]
            mask_seq = torch.tensor(mask_seq, dtype=torch.float32)                # [seq_len]
            batch_masks  = torch.tensor(np.ones(label_seq.shape[1:]),dtype=torch.float32)

            self.sequences.append((input_seq, label_seq, batch_masks, mask_seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]

from tqdm import tqdm
import torch.optim as optim


def get_window(df):
  get_windows = [df.iloc[i:i+512].values for i in range(0, len(df), 512)]
  if len(get_windows[-1]) < 512:

    padding = np.zeros(512 - len(get_windows[-1]), dtype=get_windows[-1].dtype)
    get_windows[-1] = np.concatenate([get_windows[-1], padding])

  return np.stack(get_windows)
import time
# Supondo que você já tenha instanciado seu Dataset
for taxa in [10,20,30]:
      start_time = time.time()

      input_path=f'../imputation_Statistics/experiments/{taxa}/512/treino/input'
      label_path=f'../imputation_Statistics/experiments/{taxa}/512/treino/label'


      # Instanciar o Dataset
      dataset = ImputationDataset(
          input_path=input_path,
          label_path=label_path,
          seq_len=512,
          stride=512  # Stride igual ao tamanho da janela para janelas não sobrepostas
      )

      # Criar o DataLoader
      dataloader = DataLoader(
          dataset,
          batch_size=1024,        # Ajuste conforme a capacidade da sua GPU
          shuffle=True,
          num_workers=1         # Ajuste conforme sua máquina
      )



      
      model = MOMENTPipeline.from_pretrained(
          "AutonLab/MOMENT-1-large",
          model_kwargs={'task_name': 'reconstruction'} # For imputation, we will load MOMENT in `reconstruction` mode
          # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
      )
      # takes in tensor of shape [batch_size, n_channels, context_length]
      import torch

      # ... (your existing code) ...

      # Check the number of available GPUs and set the device accordingly
      num_gpus = torch.cuda.device_count()
      if num_gpus > 0:
          # Use the first GPU if available
          device = torch.device("cuda:1")
          print(f"Using GPU: {torch.cuda.get_device_name(device)}")
      else:
          device = torch.device("cpu")
          print("Using CPU")
    

      criterion = torch.nn.MSELoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

      model = model.to(device).float()
      model.train()

      for epoch in range(10): #fine tuning
            # Iterar sobre o DataLoader e inspecionar um batch
            for batch_idx, (input_seq, label_seq,batch_masks, mask_seq) in enumerate(dataloader):
                
                input_seq = input_seq.to(device).requires_grad_()            # [batch_size, 1, seq_len]
                label_seq = label_seq.to(device)

                batch_masks = batch_masks.to(device).long()

                # Randomly mask some patches of data
                mask = mask_seq.to(device).long()
                output = model(x_enc=input_seq, input_mask=batch_masks, mask=mask)

                # Compute loss
                recon_loss = criterion(output.reconstruction, label_seq)
                observed_mask = batch_masks * (1 - mask)
                masked_loss = observed_mask * recon_loss

                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)
                
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch} loss: {loss.item()}")
 
      end_time = time.time()
      print(taxa ,end_time - start_time)
      #test

      input_path=f'../imputation_Statistics/experiments/{taxa}/512/teste/input'
      label_path=f'../imputation_Statistics/experiments/{taxa}/512/teste/label'

      series = os.listdir(input_path)

      model.eval()
      for s in series:
          df_input = pd.read_csv(os.path.join(input_path, s), index_col=0)
          df_label = pd.read_csv(os.path.join(label_path, s), index_col=0)
          preds = []
          for col in df_input.columns:
              series_input = df_input[col].reset_index(drop=True)
              series_label = df_label[col].reset_index(drop=True)
              std = max(series_label.std(),1e-10)
              mean_ = series_label.mean()

              series_label = (series_label - series_label.mean()) / max(series_label.std(),1e-10)

              windows_input = get_window(series_input)
              windows_label = get_window(series_label)
              masks = np.ones((len(windows_label),512))
              mask = np.stack([np.isnan(i).astype(int) for i in windows_input])

              batch_x = torch.tensor(windows_label, dtype=torch.float32).unsqueeze(1)
              masks = torch.tensor(masks, dtype=torch.float32)
              mask = torch.tensor(mask, dtype=torch.float32)
              batch_x = batch_x.to(device).float()
              masks = masks.to(device).long()
              mask = mask.to(device).long()

              output = model(x_enc=batch_x, input_mask=masks, mask=mask)
              preds.append(output.reconstruction.detach().cpu().numpy().reshape(-1, 1)*std +mean_)

          # Concatenate the predictions for each column horizontally to form the final DataFrame
          data_iputed = pd.DataFrame(np.concatenate(preds, axis=1), columns=df_label.columns)
          data_iputed = data_iputed.iloc[:df_label.shape[0],:]

          data_iputed.reset_index(drop=True, inplace=True)
          data_iputed.index = df_label.index # Assign the original index
          data_iputed.to_csv(f'../imputation_Statistics/experiments/{taxa}/512/teste/imputed/moment_{s}')

      input_path=f'../imputation_Statistics/experiments/{taxa}/512/treino/input'
      label_path=f'../imputation_Statistics/experiments/{taxa}/512/treino/label'

      series = os.listdir(input_path)


    #   for s in series:
    #       df_input = pd.read_csv(os.path.join(input_path, s), index_col=0)
    #       df_label = pd.read_csv(os.path.join(label_path, s), index_col=0)
    #       preds = []
    #       for col in df_input.columns:
    #           series_input = df_input[col].reset_index(drop=True)
    #           series_label = df_label[col].reset_index(drop=True)
    #           std = max(series_label.std(),1e-10)
    #           mean_ = series_label.mean()

    #           series_label = (series_label - series_label.mean()) / max(series_label.std(),1e-10)

    #           windows_input = get_window(series_input)
    #           windows_label = get_window(series_label)
    #           masks = np.ones((len(windows_label),512))
    #           mask = np.stack([np.isnan(i).astype(int) for i in windows_input])

    #           batch_x = torch.tensor(windows_label, dtype=torch.float32).unsqueeze(1)
    #           masks = torch.tensor(masks, dtype=torch.float32)
    #           mask = torch.tensor(mask, dtype=torch.float32)
    #           batch_x = batch_x.to(device).float()
    #           masks = masks.to(device).long()
    #           mask = mask.to(device).long()

    #           output = model(x_enc=batch_x, input_mask=masks, mask=mask)
            #   preds.append(output.reconstruction.detach().cpu().numpy().reshape(-1, 1)*std +mean_)

        #   # Concatenate the predictions for each column horizontally to form the final DataFrame
        #   data_iputed = pd.DataFrame(np.concatenate(preds, axis=1), columns=df_label.columns)
        #   data_iputed = data_iputed.iloc[:df_label.shape[0],:]

        #   data_iputed.reset_index(drop=True, inplace=True)
        #   data_iputed.index = df_label.index # Assign the original index
        #   data_iputed.to_csv(f'../imputation_Statistics/experiments/{taxa}/512/treino/imputed/moment_{s}')
