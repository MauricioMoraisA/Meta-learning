# 1. Importação de Bibliotecas Necessárias

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 2. Funções Utilitárias
# 2.1. Função para Obter Todos os Caminhos de Entrada

def get_all_input_paths(base_path):
    input_paths = []
    # Caminhar recursivamente pelas pastas
    for root, dirs, files in os.walk(base_path):
        if "imputed" in root:
            input_paths.append(root)
    return input_paths

# Obter os diretórios de experimentos
dirs = np.sort(get_all_input_paths('./imputation_Statistics/experiments'))
experiment_dirs = []
for experiment in dirs:
    base = experiment.split('/')
    experiment_dirs.append("/".join(base[:5]))
experiment_dirs = list(np.unique(experiment_dirs))

# 2.2. Função para Calcular Valores Mínimos e Máximos para Normalização
def compute_min_max_values(experiment_dirs, wind):
    """
    Computa o mínimo e máximo globais para normalizar os dados de entrada.
    """
    all_train_data = []

    for base_dir in experiment_dirs:
        # Carregar os dados de treino para calcular o min e o max
        train_file = os.path.join(base_dir, 'treino', 'imputed', 'data_full_processed.csv')
        if not os.path.exists(train_file):
            continue  # Pular se o arquivo não existir
        train_data = pd.read_csv(train_file)

        # Obter apenas as colunas de entrada (val_real_*)
        input_columns = [f"val_real_{i}" for i in range(wind)]
        available_columns = train_data.columns.tolist()
        missing_columns = set(input_columns) - set(available_columns)
        if missing_columns:
            print(f"Colunas faltantes no experimento {base_dir}: {missing_columns}")
            continue  # Pular este experimento se as colunas não existirem

        train_inputs = train_data[input_columns].values

        all_train_data.append(train_inputs)

    if len(all_train_data) == 0:
        raise ValueError("Nenhum dado de treinamento encontrado para calcular min/max.")

    # Concatenar os dados de todos os experimentos (neste caso, apenas um)
    all_train_data = np.vstack(all_train_data)

    # Calcular o mínimo e o máximo global (para todas as características)
    min_values = np.nanmin(all_train_data, axis=0)
    max_values = np.nanmax(all_train_data, axis=0)

    return min_values, max_values

class ExperimentDataset(Dataset):
    def __init__(self, base_dir, mode='treino', wind=48, min_values=None, max_values=None):
        """
        Inicializa o dataset lendo os dados da pasta base_dir, aplica a normalização Min-Max e realiza o balanceamento das classes.
        """
        # Verificar se estamos no modo de treino ou teste
        assert mode in ['treino', 'teste'], "O modo precisa ser 'treino' ou 'teste'"

        # Montar o caminho correto para treino ou teste
        data_file = os.path.join(base_dir, mode, 'imputed', 'data_full_processed.csv')

        # Verificar se o arquivo existe
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"O arquivo {data_file} não foi encontrado.")

        # Carregar o CSV
        self.data = pd.read_csv(data_file)
        self.data = self.data.reset_index(drop=True)  # Resetar o índice para garantir que seja sequencial

        # Definir as colunas de entrada e as multilabels
        self.input_columns = [f"val_real_{i}" for i in range(wind)]  # As primeiras 'wind' colunas são os dados de entrada
        self.multiclass_column = 'multiclass'  # Coluna multiclass
        self.multilabel_columns = [col for col in self.data.columns if 'multabel' in col]  # Colunas multilabel

        # Preencher NaN com a média da coluna nos dados de entrada
        for col in self.input_columns:
            # mean_value = self.data[col].mean()
            self.data[col] = self.data[col].fillna(-1)

        # Normalizar os dados de entrada usando Min-Max (se min_values e max_values forem fornecidos)
        if min_values is not None and max_values is not None:
            for i, col in enumerate(self.input_columns):
                self.data[col] = (self.data[col] - min_values[i]) / (max_values[i] - min_values[i] + 1e-5)  # Para evitar divisão por zero

        # Garantir que os rótulos multiclass sejam inteiros
        self.data[self.multiclass_column] = self.data[self.multiclass_column].astype(int)

        # Obter os rótulos únicos e ajustar para começar em 0
        unique_labels = self.data[self.multiclass_column].unique()
        min_label = unique_labels.min()
        max_label = unique_labels.max()
        self.data[self.multiclass_column] -= min_label  # Ajustar os rótulos para começar em 0

        # Atualizar o número de classes
        self.n_multiclass_classes = max_label - min_label + 1

        if mode == 'treino':
                # Contar o número de amostras por classe
                class_counts = self.data[self.multiclass_column].value_counts()

                # Realizar data augmentation para classes com menos de 30 amostras
                for cls in class_counts.index:
                    cls_count = class_counts[cls]
                    if cls_count < 30 and cls_count > 0:
                        # Selecionar os dados da classe
                        cls_data = self.data[self.data[self.multiclass_column] == cls]

                        # Inverter as séries temporais nas colunas de entrada para criar a versão espelhada
                        cls_data_inverted = cls_data.copy()
                        cls_data_inverted[self.input_columns] = cls_data_inverted[self.input_columns].apply(lambda row: row[::-1], axis=1)

                        # Concatenar os dados originais com os dados invertidos
                        self.data = pd.concat([self.data, cls_data_inverted], axis=0, ignore_index=True)

                # Após o data augmentation, contar novamente o número de amostras por classe
                class_counts = self.data[self.multiclass_column].value_counts()
                print(f'Dataset aumentado{self.data.shape}')

                # Determinar o número máximo de amostras permitido por classe
                # Você pode escolher um valor específico ou usar, por exemplo, o percentil 90
                max_samples_per_class = int(class_counts.quantile(0.8))
                print(f"Limite máximo de amostras por classe: {max_samples_per_class}")

                # Reduzir o número de amostras das classes majoritárias
                balanced_data = []
                for cls in class_counts.index:
                    cls_data = self.data[self.data[self.multiclass_column] == cls]

                    if len(cls_data) > max_samples_per_class:
                        # Realizar undersampling aleatório para reduzir o número de amostras
                        cls_data = cls_data.sample(n=max_samples_per_class, random_state=42)

                    balanced_data.append(cls_data)

                # Concatenar todas as classes balanceadas
                self.data = pd.concat(balanced_data, ignore_index=True)
                self.data[self.multiclass_column] = self.data[self.multiclass_column].astype(int)
                print(f'Datset Balaceado{self.data.shape}')

                # Embaralhar os dados
                self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            # Se estiver no modo de teste, não balancear as classes
            self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retorna os dados de entrada e os alvos para o índice dado.
        """
        # Dados de entrada
        input_data = self.data.iloc[idx][self.input_columns].values.astype(np.float32)
        input_data = input_data[:, np.newaxis]  # Adicionar dimensão de canal

        # Multilabel alvo (usada no treinamento)
        multilabel_target = self.data.iloc[idx][self.multilabel_columns].values.astype(np.float32)

        # Multiclass alvo (usada no teste)
        multiclass_target = int(self.data.iloc[idx][self.multiclass_column])

        return torch.tensor(input_data), torch.tensor(multilabel_target), torch.tensor(multiclass_target)

# 4. Função para Carregar Dados de um Único Experimento
from torch.utils.data import random_split, Subset

def load_single_experiment_data(experiment_dir, wind, batch_size, min_values=None, max_values=None, val_split=0.1, random_seed=42):
    # Criar o dataset completo de treinamento
    full_train_dataset = ExperimentDataset(experiment_dir, mode='treino', wind=wind, min_values=min_values, max_values=max_values)
    
    # Determinar o tamanho do conjunto de validação
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    # Definir a seed para reprodutibilidade
    generator = torch.Generator().manual_seed(random_seed)
    
    # Dividir o dataset em treino e validação
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    
    # Criar o dataset de teste
    test_dataset = ExperimentDataset(experiment_dir, mode='teste', wind=wind, min_values=min_values, max_values=max_values)
    
    # Criar data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

class HybridLSTM(nn.Module):
    def __init__(self, seq_len, n_multilabel_classes, n_multiclass_classes, filters=64, filterslstm=30,
                 num_rnn_layers=2, num_conv_layers=6, kernel_size=2, dense_neurons=80, dp=0.5):
        super(HybridLSTM, self).__init__()

        self.rnn_layers = nn.ModuleList()

        # Primeira camada LSTM bidirecional
        self.rnn_layers.append(
            nn.LSTM(input_size=1, hidden_size=filterslstm, batch_first=True, bidirectional=True,
                    dropout=dp if num_rnn_layers > 1 else 0.0)
        )

        # LSTMs subsequentes unidirecionais
        for i in range(1, num_rnn_layers):
            # A saída da primeira camada bidirecional terá 2 * hidden_size
            self.rnn_layers.append(
                nn.LSTM(input_size=filterslstm * 2, hidden_size=filterslstm, batch_first=True, bidirectional=False,
                        dropout=dp if num_rnn_layers > 1 else 0.0)
            )

        # Ajustando o tamanho da camada de normalização
        self.layer_norm_lstm = nn.LayerNorm(filterslstm)

        # Definindo as camadas convolucionais
        conv_layers = []
        for i in range(num_conv_layers):
            in_channels = 1 if i == 0 else filters
            conv_layers += [
                nn.Conv1d(in_channels, filters, kernel_size, stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(filters),
                nn.ReLU(inplace=True),
                nn.Dropout(dp)
            ]
        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculando o tamanho da saída convolucional corretamente
        self.conv_output_size = self._get_conv_output_size(seq_len)

        # Camadas densas após convolução
        self.dense_conv = nn.Linear(self.conv_output_size, dense_neurons)
        self.act_dense_conv = nn.ReLU(inplace=True)

        # Definindo o tamanho de saída da última LSTM (unidirecional)
        lstm_output_size = filterslstm * seq_len

        # Concatenando as saídas de LSTM e convolução
        final_input_size = lstm_output_size + dense_neurons

        # Camadas de saída
        self.multilabel_output = nn.Linear(final_input_size, n_multilabel_classes)
        self.multiclass_output = nn.Linear(final_input_size, n_multiclass_classes)

    def _get_conv_output_size(self, seq_len):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_len)  # (batch_size, num_channels, sequence_length)
            output = self.conv_layers(dummy_input)
            return output.numel()  # Número total de elementos na saída convolucional

    def forward(self, x):
        # Caminho convolucional
        x_conv = x.clone().permute(0, 2, 1)  # Mudando para (batch_size, num_canais, comprimento)
        x_conv = self.conv_layers(x_conv)
        x_conv = x_conv.view(x_conv.size(0), -1)  # Achatar a saída convolucional
        x_conv = self.act_dense_conv(self.dense_conv(x_conv))

        # Caminho LSTM
        lstm_out, _ = self.rnn_layers[0](x)  # Primeira LSTM bidirecional
        for rnn in self.rnn_layers[1:]:
            lstm_out, _ = rnn(lstm_out)  # LSTMs subsequentes unidirecionais

        # Normalizando a saída da última LSTM
        lstm_out = self.layer_norm_lstm(lstm_out)
        lstm_out = lstm_out.contiguous().view(lstm_out.size(0), -1)  # Achatar a saída LSTM

        # Concatenando as saídas LSTM e convolucional
        x_concat = torch.cat([lstm_out, x_conv], dim=1)

        # Saídas finais
        return self.multilabel_output(x_concat), self.multiclass_output(x_concat)

# 6. Definição da Classe EarlyStopping
# Função de early stopping

class EarlyStopping:
    def __init__(self, patience=7, delta=0.1, path='checkpoint/models/best_model.pt', check_interval=5):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.check_interval = check_interval  # Intervalo para verificação
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.Inf

    def __call__(self, val_loss, model, epoch):
        if epoch % self.check_interval != 0:
            return  # Não verifica em épocas que não são múltiplas do intervalo

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)  # Salva o modelo no início
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)  # Salva o modelo quando há melhora
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        directory = os.path.dirname(self.path)
        if directory != '':
            os.makedirs(directory, exist_ok=True)
        
        torch.save(model.state_dict(), self.path)
        print(f'Modelo salvo! Perda de validação melhorada: {val_loss:.4f}')
        self.best_loss = val_loss

# 7. Configuração do Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 8. Loop Principal de Treinamento
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

# Loop Principal de Treinamento
import time
for experiment_dir in experiment_dirs:
    start  =time.time()
    torch.cuda.empty_cache()
    wind = int(experiment_dir.split('/')[-1])

    batch_size = 1024# Ajuste conforme necessário
    
    # Verificar se os arquivos de dados existem
    train_file = os.path.join(experiment_dir, 'treino', 'imputed', 'data_full_processed.csv')
    test_file = os.path.join(experiment_dir, 'teste', 'imputed', 'data_full_processed.csv')
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Dados não encontrados para o experimento: {experiment_dir}")
        continue

    # Calcular valores mínimos e máximos para normalização usando apenas o experimento atual
    min_values, max_values = compute_min_max_values([experiment_dir], wind=wind)

    # Carregar DataLoaders de treino, validação e teste
    train_loader, val_loader, test_loader = load_single_experiment_data(
        experiment_dir, wind=wind, batch_size=batch_size,
        min_values=min_values, max_values=max_values,
        val_split=0.3, random_seed=42
    )

    # Obter o número de classes
    n_multilabel_classes = train_loader.dataset.dataset.n_multiclass_classes
    n_multiclass_classes = train_loader.dataset.dataset.n_multiclass_classes

    print(f"Experimento: {experiment_dir}")
    print("Número de classes multiclass:", n_multiclass_classes)
    print("Número de classes multilabel:", n_multilabel_classes)

    # Instanciar o modelo com o número correto de classes
    model = HybridLSTM(
        seq_len=wind,
        n_multilabel_classes=n_multilabel_classes,
        n_multiclass_classes=n_multiclass_classes
    ).to(device)

    model.apply(initialize_weights)
    # model = nn.DataParallel(model)


    #  Calculando o número total de parâmetros
    total_params = sum(p.numel() for p in model.parameters())

    # Imprimindo o número total de parâmetros
    print(f"Total de parâmetros do modelo: {total_params}")

    # Definir as funções de perda
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced',
                                         classes = np.unique(train_loader.dataset.dataset.data['multiclass']),
                                         y = train_loader.dataset.dataset.data['multiclass'])
    class_weights = torch.tensor(class_weights,dtype=torch.float32,device=device)
    criterion_multilabel = nn.BCEWithLogitsLoss().to(device)  # Usando BCEWithLogitsLoss para maior estabilidade
    criterion_multiclass = nn.CrossEntropyLoss(weight=class_weights).to(device)  # Para classificação multiclass

    # Otimizador
    initial_learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=1e-5)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, verbose=True, min_lr=1e-7)

    # Diretório para salvar os modelos
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    # Early Stopping com caminho atualizado
    model_path = os.path.join(model_dir, f"best_model_{experiment_dir.replace('/', '_')}_wind_{wind}.pt")
    early_stopping = EarlyStopping(patience=30, delta=0.01, path=model_path)


    # Loop de Treinamento
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        multilabel_outputs_epoch = []
        multilabel_targets_epoch = []

        for batch_x, batch_y_multilabel, batch_y_multiclass in train_loader:
            batch_x = batch_x.to(device)
            batch_y_multilabel = batch_y_multilabel.to(device)
            batch_y_multiclass = batch_y_multiclass.to(device)

            optimizer.zero_grad()
            outputs_multilabel, outputs_multiclass = model(batch_x)

            # Calcular as perdas
            loss_multilabel = criterion_multilabel(outputs_multilabel, batch_y_multilabel)
            loss_multiclass = criterion_multiclass(outputs_multiclass, batch_y_multiclass)

            # Perda total
            total_loss_batch =  0.4 *loss_multilabel + 0.6*loss_multiclass

            total_loss_batch.backward()
            optimizer.step()
            total_loss += total_loss_batch.item()

            # Coletar saídas e alvos para calcular o F1 Score
            multilabel_outputs_epoch.append(outputs_multilabel.detach().cpu())
            multilabel_targets_epoch.append(batch_y_multilabel.detach().cpu())

        # Após o loop dos batches, calcular o F1 Score para a época
        multilabel_outputs_epoch = torch.cat(multilabel_outputs_epoch)
        multilabel_targets_epoch = torch.cat(multilabel_targets_epoch)

        # Aplicar sigmoide nas saídas
        multilabel_probs_epoch = torch.sigmoid(multilabel_outputs_epoch)

        # Binarizar as saídas com um limiar de 0.5
        multilabel_preds_epoch = (multilabel_probs_epoch >= 0.6).float()

        # Calcular o F1 Score macro
        multilabel_f1_score_train = f1_score(multilabel_targets_epoch.numpy(), multilabel_preds_epoch.numpy(), average='macro')

        # Validação
        model.eval()
        valid_loss = 0
        correct_multiclass = 0
        total_multiclass = 0
        multilabel_outputs_val = []
        multilabel_targets_val = []

        with torch.no_grad():
            for batch_x, batch_y_multilabel, batch_y_multiclass in val_loader:
                batch_x = batch_x.to(device)
                batch_y_multilabel = batch_y_multilabel.to(device)
                batch_y_multiclass = batch_y_multiclass.to(device)
                outputs_multilabel, outputs_multiclass = model(batch_x)

                # Perda de validação para multilabel
                loss_multilabel = criterion_multilabel(outputs_multilabel, batch_y_multilabel)
                valid_loss += loss_multilabel.item()

                # Coletar saídas e alvos para calcular o F1 Score
                multilabel_outputs_val.append(outputs_multilabel.cpu())
                multilabel_targets_val.append(batch_y_multilabel.cpu())

                # Precisão multiclass
                _, predicted_multiclass = torch.max(outputs_multiclass, 1)
                correct_multiclass += (predicted_multiclass == batch_y_multiclass).sum().item()
                total_multiclass += batch_y_multiclass.size(0)

        valid_loss /= len(val_loader)
        accuracy_multiclass = 100 * correct_multiclass / total_multiclass

        # Calcular o F1 Score para a validação
        multilabel_outputs_val = torch.cat(multilabel_outputs_val)
        multilabel_targets_val = torch.cat(multilabel_targets_val)

        # Aplicar sigmoide nas saídas
        multilabel_probs_val = torch.sigmoid(multilabel_outputs_val)

        # Binarizar as saídas com um limiar de 0.5
        multilabel_preds_val = (multilabel_probs_val >= 0.6).float()

        # Calcular o F1 Score macro
        multilabel_f1_score_val = f1_score(multilabel_targets_val.numpy(), multilabel_preds_val.numpy(), average='macro',zero_division=1)

        if epoch % 15 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, "
                f"Validation Loss: {valid_loss:.4f}, Multiclass Accuracy: {accuracy_multiclass:.2f}%, "
                f"Train Multilabel F1 Score: {multilabel_f1_score_train:.4f}, "
                f"Validation Multilabel F1 Score: {multilabel_f1_score_val:.4f}")

        # Passo do Scheduler
        scheduler.step(valid_loss)

        # Early Stopping
        early_stopping(valid_loss, model,epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Carregar o melhor modelo salvo pelo early stopping
    model.load_state_dict(torch.load(model_path))

    #Avaliação Final no Conjunto de Teste

    model.eval()
    test_loss = 0
    correct_multiclass_test = 0
    total_multiclass_test = 0
    multilabel_outputs_test = []
    multilabel_targets_test = []

    with torch.no_grad():
        for batch_x, batch_y_multilabel, batch_y_multiclass in test_loader:
            batch_x = batch_x.to(device)
            batch_y_multilabel = batch_y_multilabel.to(device)
            batch_y_multiclass = batch_y_multiclass.to(device)
            outputs_multilabel, outputs_multiclass = model(batch_x)

            # Perda de teste para multilabel
            loss_multilabel = criterion_multilabel(outputs_multilabel, batch_y_multilabel)
            test_loss += loss_multilabel.item()

            # Coletar saídas e alvos para calcular o F1 Score
            multilabel_outputs_test.append(outputs_multilabel.cpu())
            multilabel_targets_test.append(batch_y_multilabel.cpu())

            # Precisão multiclass
            _, predicted_multiclass = torch.max(outputs_multiclass, 1)
            correct_multiclass_test += (predicted_multiclass == batch_y_multiclass).sum().item()
            total_multiclass_test += batch_y_multiclass.size(0)

    test_loss /= len(test_loader)
    accuracy_multiclass_test = 100 * correct_multiclass_test / total_multiclass_test

    # Calcular o F1 Score para o teste
    multilabel_outputs_test = torch.cat(multilabel_outputs_test)
    multilabel_targets_test = torch.cat(multilabel_targets_test)

    # Aplicar sigmoide nas saídas
    multilabel_probs_test = torch.sigmoid(multilabel_outputs_test)

    # Binarizar as saídas com um limiar
    multilabel_preds_test = (multilabel_probs_test >= 0.6).float()

    # Calcular o F1 Score macro
    multilabel_f1_score_test = f1_score(multilabel_targets_test.numpy(), multilabel_preds_test.numpy(), average='macro')

    print(f"Teste Loss: {test_loss:.4f}, Multiclass Test Accuracy: {accuracy_multiclass_test:.2f}%, "
        f"Multilabel Test F1 Score: {multilabel_f1_score_test:.4f}")

    