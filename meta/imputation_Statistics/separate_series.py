import os
import os.path
import pandas as pd
import numpy as np

# Função para inserir valores faltantes no DataFrame
def inserir_faltosos_por_janela(df, janela, taxa_faltantes):
    """
    Insere valores faltantes (NaNs) em cada janela de cada coluna de um DataFrame com base em uma taxa especificada.

    Parâmetros:
    df (pd.DataFrame): O DataFrame com as colunas a serem modificadas.
    janela (int): O tamanho da janela para inserir os valores faltantes.
    taxa_faltantes (float): A porcentagem de valores faltantes a serem inseridos em cada janela (0 a 100).

    Retorna:
    pd.DataFrame: O DataFrame modificado com os valores faltantes inseridos.
    """
    # Cria uma cópia do DataFrame para evitar modificar o original
    df_modificado = df.copy(deep=True)

    # Converte a taxa de faltantes de percentual para decimal
    proporcao_faltantes = taxa_faltantes / 100

    # Percorre cada coluna do DataFrame
    for coluna in df_modificado.columns:
        # Percorre o DataFrame em janelas
        for i in range(0, df_modificado.shape[0], janela):
            # Seleciona a janela atual
            janela_atual = df_modificado[coluna].iloc[i:i + janela]

            # Número de valores faltantes a serem inseridos nesta janela
            n_faltantes = int(len(janela_atual) * proporcao_faltantes)

            # Índices aleatórios na janela para serem substituídos por NaN
            if n_faltantes > 0:
                indices_faltantes = np.random.choice(janela_atual.index, size=n_faltantes, replace=False)

                # Insere os NaNs nos índices selecionados
                df_modificado.loc[indices_faltantes, coluna] = np.nan

    return df_modificado

# Função para obter todos os arquivos CSV de uma pasta
main_folder = 'datasets'

def get_df(main_folder=main_folder):
    csv_files = []
    # Caminhar recursivamente pela pasta principal
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.endswith(".csv"):
                # Adicionar o caminho completo do arquivo CSV na lista
                csv_files.append(os.path.join(root, file))

    return csv_files

# Função principal para criar a estrutura de experimentos
def create_experiment_structure(base_path, taxa=10,holdout=1):
    # Criar a pasta raiz com a taxa atual de valores faltosos
    root_path = os.path.join(base_path, str(taxa))
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    print(f'Procesando {root_path}')

    # Obter a lista de arquivos CSV
    series = get_df()
    for serie in series:
        df = pd.read_csv(serie, index_col=0, parse_dates=True, header=0)
        nome_serie = os.path.splitext(os.path.basename(serie))[0]
    
        # Definir tamanhos das subpastas
        sizes = ["48", "512", "1024"] # pastas referentes aos tamanhos de janelamento

        for size in sizes:
            w = (df.shape[0] // int(size))
            if w < 2:
                continue

            # Criar subpasta para cada tamanho dentro da pasta raiz
            size_path = os.path.join(root_path, size)
            os.makedirs(size_path, exist_ok=True)
           
            if w == 2:
                # Se houver apenas duas janelas, uma para treino e uma para teste
                df_treino, df_teste = df.iloc[:int(size), :], df.iloc[int(size):, :]
            elif w == 3:
                # Se houver três janelas, duas para treino e uma para teste
                df_treino, df_teste = df.iloc[:int(2 *size), :], df.iloc[2*int(size):, :]  
            elif w>3:
            
                # Se houver mais de três janelas, dividir 75% das janelas para treino e 25% para teste
                n_janelas_treino = max(1, int(w * 0.75))  # Garantir pelo menos uma janela para treino
                df_treino, df_teste = df.iloc[:n_janelas_treino * int(size), :], df.iloc[n_janelas_treino * int(size):, :]
            # print(df_treino.shape,df_teste.shape)
            # Estrutura para treino e teste
            for tipo in ["treino", "teste"]:
                tipo_path = os.path.join(size_path, tipo)
                os.makedirs(tipo_path, exist_ok=True)
                
                # Criar as pastas input e label
                input_path = os.path.join(tipo_path, "input")
                label_path = os.path.join(tipo_path, "label")
                os.makedirs(input_path, exist_ok=True)
                os.makedirs(label_path, exist_ok=True)

                # Definir qual dataframe usar
                df_atual = df_treino if tipo == "treino" else df_teste

                # Garantir que o DataFrame não está vazio antes de salvar
                if df_atual.empty:
                    # print(f"DataFrame {tipo} para série {nome_serie} e janela {size} está vazio. Ignorando.")
                    continue

                # Aplicar a transformação ao input e salvar
                df_input = inserir_faltosos_por_janela(df_atual, int(size), taxa)
                if not df_input.empty:
                    df_input.to_csv(os.path.join(input_path, f"{holdout}_{nome_serie}.csv"))
                else:
                    # print(f"DataFrame input para série {nome_serie}, tipo {tipo}, e janela {size} está vazio após inserção de faltosos. Ignorando.")
                    pass
                # Salvar o dataframe original no label
                df_atual.to_csv(os.path.join(label_path, f"{holdout}_{nome_serie}.csv"))

# Chamar a função para criar a estrutura e salvar os dados para treino e teste com as taxas de 10, 20 e 30 % de valores faltosos, 
# por tamanho de janelas

if __name__ == "__main__":
    for  i in range(30):
     print(f'hold:{i}')
     for taxa in [10, 20, 30]:
        create_experiment_structure("experiments", taxa=taxa,holdout=i)