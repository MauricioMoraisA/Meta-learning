import sys
sys.path.append('../imputation_Statistics')
import os
import pandas as pd
import warnings
import time

from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
from utils import metodos

# Função para ler todas as pastas e fazer a imputação usando o código fornecido
def get_df(main_folder):
    csv_files = []
    # Caminhar recursivamente pela pasta principal
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".csv"):
                # Adicionar o caminho completo do arquivo CSV na lista
                csv_files.append(os.path.join(root, file))
    return csv_files

def get_all_input_paths(base_path):
    input_paths = []
    # Caminhar recursivamente pelas pastas
    for root, dirs, files in os.walk(base_path):
        if "input" in root:
            input_paths.append(root)
    return input_paths

def imputar_dados_e_salvar(base_path):
    names = ['mean', 'median', 'inter_linear', 'inter_cubic', 'inter_akima', 'inter_poly5', 'inter_spline5', 'mean_mov_3', 'backfill', 'ffill']
    pasta = 'imputed'
    imputados = []
    phats = get_all_input_paths(base_path)
  
    print('Iniciou')
    for expe in tqdm(phats, desc="Processing input files", position=0):
        
        wind = int(expe.split('/')[2]) # obter o tamanho da janela do experimento atual
        print(f'Folder: {expe}')
        series = get_df(expe)
        file = expe.replace('input', pasta)
        os.makedirs(file, exist_ok=True)

        # Caminhar pelas pastas de input geradas anteriormente
        for s in tqdm(series, desc="Processing CSV files", leave=True, position=1):
            # Tentar detectar automaticamente o delimitador e ler o arquivo corretamente
            try:
                data = pd.read_csv(s, index_col=0, parse_dates=True, header=0, sep=',', encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                data = pd.read_csv(s, index_col=0, parse_dates=True, header=0, sep=',', encoding='latin1', low_memory=False)
            except pd.errors.ParserError:
                data = pd.read_csv(s, index_col=0, parse_dates=True, header=0, sep=';', encoding='utf-8', low_memory=False)

            # Verificar se há índices duplicados
            if data.index.duplicated().any():
                print(f"Warning: The {s} file contains duplicate indexes.")

            # Salvar o índice original
            original_index = data.index

            # Substituir o índice por uma sequência estritamente crescente
            data.reset_index(drop=True, inplace=True)

            for k in range(len(names)):
                df = data.copy(deep=True) # copiar para não alterar no df real
                # Captura o tempo inicial

                start_time = time.time()
                for col in df.columns:
                    # Dividir a coluna em janelas sem sobreposição
                    for start in range(0, df.shape[0], wind):
                        end = start + wind
                        df_window = df[col].iloc[start:end]
                        if df_window.shape[0]<9:
                             df_window = df[col].iloc[start-9:end]

                        # Aplicar método de imputação à janela atual
                        if df_window.isnull().any():
                            try:
                                y_hat = metodos(df_window, k)
                            except ValueError as e:
                                print(f"Erro ao aplicar o método {names[k]} na janela de {start} a {end} na coluna {col}: {e}")
                                continue
                            
                            # Garantir que y_hat é uma Série e selecionar a coluna correta
                            if isinstance(y_hat, pd.DataFrame):
                                y_hat = y_hat.iloc[:, -1]  # Selecionar a última coluna, assumindo que seja a imputada

                            # Converter y_hat para uma Série, se necessário
                            if not isinstance(y_hat, pd.Series):
                                y_hat = pd.Series(y_hat.squeeze(), index=df_window.index)

                            # Alinhar os índices
                            y_hat.index = df_window.index
                            
                            # passar valores
                            df.loc[df_window.index, col] = y_hat
                
                # Restaurar o índice original no DataFrame imputado
                df.index = original_index
                        
                # Salvar o dataframe imputado
                output_file_path = os.path.join(file, names[k] + '_' + os.path.basename(s))
                # df.to_csv(output_file_path)
                # Captura o tempo final
            end_time = time.time()

            # Calcula o tempo de execução
            execution_time = end_time - start_time
            imputados.append([expe, k,execution_time])
    return imputados            
    print('Terminou')

if __name__ == "__main__":
    imputados = imputar_dados_e_salvar('experiments')
    print(imputados)