
import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset



class Dataloadertimes(Dataset):
    def __init__(self,root_label, root_input , taxa=10,in_out ='treino'):
        self.root_horse = root_input  # Dados com valores faltantes
        self.root_zebra = root_label  # Dados sem valores faltantes
        print(self.root_horse ,'\n', self.root_zebra)
        self.transform = A.Compose([
            A.Lambda(image=self.to_float32),
            ToTensorV2()
        ])

        # Lista para armazenar informações sobre cada janela
        self.data_index = []
        self.dataframes_horse = {}
        self.dataframes_zebra = {}
        self.means_stds = {}  # Armazenar médias e desvios padrão para cada coluna
        self.taxa = taxa
        self.in_out = in_out
        # Obter lista de arquivos comuns entre as duas pastas
        input_files = [f for f in os.listdir(root_input) if f.endswith('.csv')]
        
        label_files = [f for f in os.listdir(root_label) if f.endswith('.csv')]
        common_files = set(input_files).intersection(label_files)

        for file_name in common_files:
            file_path_input = os.path.join(root_input, file_name)
            file_path_label = os.path.join(root_label, file_name)

            # Ler os dataframes correspondentes
            df_input= pd.read_csv(file_path_input, index_col=0)
            
            df_label = pd.read_csv(file_path_label, index_col=0)

            # Armazenar os dataframes originais
            self.dataframes_horse[file_path_input] = df_input.copy()
            self.dataframes_zebra[file_path_label] = df_label.copy()

            # Preencher NaNs na 'horse' com a média da coluna
            df_input_filled = df_input.apply(lambda x: x.fillna(x.mean()), axis=0)

            # Normalizar ambos os dataframes e armazenar médias e desvios padrão
            df_horse_normalized, means_stds_input = self.normalize_dataframe(df_input_filled)
            df_zebra_normalized, means_stds_label = self.normalize_dataframe(df_label)

            # Armazenar as médias e desvios padrão
            self.means_stds[file_name] = means_stds_input  # Assumindo que ambas têm as mesmas colunas

            # Atualizar os dataframes normalizados
            self.dataframes_horse[file_path_input] = df_horse_normalized
            self.dataframes_zebra[file_path_label] = df_zebra_normalized

            # Para cada coluna
            for col in df_horse_normalized.columns:
                data_input = df_horse_normalized[col].values
                data_label = df_zebra_normalized[col].values

                # Garantir que ambos tenham o mesmo comprimento
                min_length = min(len(data_input), len(data_label))
                num_windows = min_length // 1024

                for i in range(num_windows):
                    self.data_index.append({
                        'file_input': file_path_input,
                        'file_label': file_path_label,
                        'column': col,
                        'window': i
                    })

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, index):
        info = self.data_index[index]
        file_path_horse = info['file_input']
        file_path_zebra = info['file_label']
        column = info['column']
        window_idx = info['window']

        # Recuperar os dataframes normalizados
        df_input = self.dataframes_horse[file_path_horse]
        df_label = self.dataframes_zebra[file_path_zebra]

        data_horse = df_input[column].values
        data_zebra = df_label[column].values

        start = window_idx * 1024
        end = start + 1024
        window_horse = data_horse[start:end]
        window_zebra = data_zebra[start:end]

        # Converter as janelas em imagens 32x32
        image_horse = window_horse.reshape(32, 32).astype(np.float32)
        image_zebra = window_zebra.reshape(32, 32).astype(np.float32)

        # Aplicar transformações (se houver)
        if self.transform:
            augmented_input = self.transform(image=image_horse)
            image_input = augmented_input['image']
            augmented_label = self.transform(image=image_zebra)
            image_label = augmented_label['image']

        # Armazenar informações necessárias para reescalar posteriormente
        min_ = self.means_stds[os.path.basename(file_path_horse)][column]['min_']
        k = self.means_stds[os.path.basename(file_path_horse)][column]['k']

        # Retornar o par de imagens e os parâmetros de reescalonamento
        return image_input, image_label, min_, k, info

    @staticmethod
    def to_float32(image, **kwargs):
        return image.astype(np.float32)

    @staticmethod
    def normalize_column(x):
    
        normalized = (x -x.min())/ max(1,(x.max() - x.min()))
    
        return normalized, x.min(),   max(1,(x.max() - x.min()))

    def normalize_dataframe(self, df):
        mins_k = {}
        normalized_columns = {}  # Dicionário para armazenar colunas normalizadas
        for col in df.columns:
            normalized_col, mean, std = self.normalize_column(df[col])
            normalized_columns[col] = normalized_col
            mins_k[col] = {'min_': mean, 'k': std}
        df_normalized = pd.DataFrame(normalized_columns, index=df.index)
        return df_normalized, mins_k

    def reconstruct_dataframe(self, file_name, imputed_data):
        """
        Reconstrói o dataframe imputado com os dados reescalados e salva em um arquivo CSV.
        """
        # Recuperar o dataframe original com valores faltantes
        file_path_horse = os.path.join(self.root_horse, file_name)
        df_original = self.dataframes_horse[file_path_horse]

        # Recuperar as médias e desvios padrão
        means_stds = self.means_stds[file_name]

        # Reconstruir o dataframe
        df_reconstructed = pd.DataFrame(index=df_original.index)

        # Substituir os valores faltantes pelos imputados e reescalar
        for col in df_original.columns:
            min = means_stds[col]['min_']
            k = means_stds[col]['k']
            if k == 0:
                reescaled_data = imputed_data[col] + min
            else:
                reescaled_data = (imputed_data[col] * k) + min

            # Preencher o que foi cortado com zeros ou cortar o excesso
            difference = len(df_reconstructed.index) - len(reescaled_data)
            if difference > 0:
                reescaled_data = np.concatenate([reescaled_data, np.zeros(difference)])
            elif difference < 0:
                reescaled_data = reescaled_data[:len(df_reconstructed.index)]

            # Inserir os dados reescalados no dataframe
            df_reconstructed[col] = reescaled_data

        # Salvar o dataframe reconstruído em um arquivo CSV
        output_directory = '../imputation_Statistics/experiments/'+ f'{self.taxa}/1024/{self.in_out}/imputed' 
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, f'pix_{file_name}')
        df_reconstructed.to_csv(output_path, index=True)

        print(f"Dataframe imputado salvo em: {output_path}")



class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_downsampling: bool = True,
        add_activation: bool = True,
        **kwargs
    ):
        super().__init__()
        if is_downsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    padding_mode="reflect",
                    **kwargs
                ),
                nn.InstanceNorm2d(
                    out_channels,
                    affine=False,  # Ou affine=True, se necessário
                    track_running_stats=False  # Definir como False
                ),
                nn.ReLU() if add_activation else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    **kwargs
                ),
                nn.InstanceNorm2d(
                    out_channels,
                    affine=False,  # Ou affine=True, se necessário
                    track_running_stats=False  # Definir como False
                ),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
                nn.Dropout(0.4),
            )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        """
        In a residual block, the use of two ConvBlock instances with one having
        an activation function and the other not is a design choice that promotes
        the learning of residual information.

        The purpose of a residual block is to learn the residual mapping between
        the input and output of the block. The first ConvBlock in the sequence,
        which includes an activation function, helps in capturing and extracting
        important features from the input. The activation function introduces
        non-linearity, allowing the network to model complex relationships
        between the input and output.

        The second ConvBlock does not include an activation function.
        It mainly focuses on adjusting the dimensions (e.g., number of channels)
        of the features extracted by the first ConvBlock. The absence of an
        activation function in the second ConvBlock allows the block to learn
        the residual information. By directly adding the output of the second
        ConvBlock to the original input, the block learns to capture the
        residual features or changes needed to reach the desired output.

        (Information and explanation above generated by ChatGPT)
        """
        super().__init__()
        self.block = nn.Sequential(
            ConvolutionalBlock(channels, channels, add_activation=True, kernel_size=3, padding=1),
            ConvolutionalBlock(channels, channels, add_activation=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        This skip connection, achieved through the addition operation, helps
        in propagating gradients during training and alleviates the vanishing
        gradient problem. It also facilitates the flow of information from earlier
        layers to later layers, allowing the network to learn more effectively.

        (Information and explanation above generated by ChatGPT)
        """
        return x + self.block(x)
class Generator(nn.Module):
    def __init__(
        self, img_channels: int, num_features: int = 64, num_residuals: int = 9
    ):
        """
        Generator consists of 2 layers of downsampling/encoding layer,
        followed by 9 residual blocks for 128 × 128 training images
        and then 3 upsampling/decoding layer.

        The network with 6 residual blocks can be written as:
        c7s1–64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64, and c7s1–3.

        The network with 9 residual blocks consists of:
        c7s1–64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, and c7s1–3.
        """
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),

            nn.ReLU(inplace=True),
        )

        self.downsampling_layers = nn.ModuleList(
            [
                ConvolutionalBlock(
                    num_features,
                    num_features * 2,
                    is_downsampling=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                ConvolutionalBlock(
                    num_features * 2,
                    num_features * 4,
                    is_downsampling=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.outactivation = nn.Softplus()
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.upsampling_layers = nn.ModuleList(
            [
                ConvolutionalBlock(
                    num_features * 4,
                    num_features * 2,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvolutionalBlock(
                    num_features * 2,
                    num_features * 1,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last_layer = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.downsampling_layers:
            x = layer(x)
        x = self.residual_layers(x)
        for layer in self.upsampling_layers:
            x = layer(x)
        return  self.outactivation(self.last_layer(x))

class ConvInstanceNormLeakyReLUBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        Class object initialization for Convolution-InstanceNorm-LeakyReLU layer

        We use leaky ReLUs with a slope of 0.2.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),

            nn.LeakyReLU(0.2, inplace=True),
            
         
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        """
        Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with
        k filters and stride 2. Discriminator architecture is: C64-C128-C256-C512.

        After the last layer, we apply a convolution to produce a 1-dimensional
        output.

        We use leaky ReLUs with a slope of 0.2.
        """
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),

            nn.LeakyReLU(0.2, inplace=True),
       
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                ConvInstanceNormLeakyReLUBlock(
                    in_channels,
                    feature,
                    stride=1 if feature == features[-1] else 2,
                )
            )
            in_channels = feature

        # After the last layer, we apply a convolution to produce a 1-dimensional output
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)

        # feed the model output into a sigmoid function to make a 1/0 label
        return torch.sigmoid(self.model(x))


class ToFloat32:
    def __call__(self, image, **kwargs):
        return image.float()
    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Inicializa o early stopping.
        :param patience: Número de épocas para esperar após a última melhora.
        :param min_delta: Mínima diferença absoluta entre as perdas para ser considerada uma melhora.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            # print(f"EarlyStopping: {self.counter} de {self.patience} sem melhora.")
            if self.counter >= self.patience:
                self.early_stop = True
