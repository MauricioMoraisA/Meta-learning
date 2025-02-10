import pandas as pd
import numpy as np
import os

methods =  ['mean', 'median', 'inter_linear', 'inter_cubic', 'inter_akima', 'inter_poly5', 'inter_spline5', 
            'mean_mov_3', 'backfill', 'ffill','pix','moment']

def get_all_input_paths(base_path):
    input_paths = []
    # Caminhar recursivamente pelas pastas
    for root, dirs, files in os.walk(base_path):
        if "imputed" in root:
            input_paths.append(root)
    return input_paths
dirs = np.sort(get_all_input_paths('./imputation_Statistics/experiments'))
import pandas as pd
import numpy as np

# Função personalizada para ASMAPE com tratamento de zeros

def asmape(y_true, y_pred):
    # Converte para arrays numpy se forem listas
    if isinstance(y_true, list) or isinstance(y_pred, list):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Cria uma máscara para excluir valores onde (y_true + y_pred) é zero ou NaN
    mask = (np.abs(y_true) + np.abs(y_pred)) != 0
    y_true, y_pred = y_true[mask], y_pred[mask]  # Aplicando a máscara

    # Calcula o comprimento do array filtrado sem NaNs
    len_ = np.count_nonzero(~np.isnan(y_pred))


    if len_ == 0:
        return 200  # Retorna NaN se não houver valores válidos para o cálculo

    # Calcula o ASMAPE com os valores válidos
    tmp = 100 * (np.nansum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) / len_)

    return tmp



# Função personalizada para MAPE usando nansum e evitando divisões por zero
def mape_nansum(y_true, y_pred):
    # Evitar divisão por zero (y_true == 0)
    mask = (y_true != 0) & ~np.isnan(y_true)  # Apenas valores válidos e não zero
    if np.sum(mask) == 0:
        return np.nan  # Retorna NaN se não houver valores válidos
    # Calcular o MAPE ignorando NaN e divisões por zero
    mape_value = 100 * np.nansum(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) / np.sum(mask)
    return mape_value
# Função personalizada para MAE usando nansum
def mae_nansum(y_true, y_pred):
    mae_value = np.nansum(np.abs(y_true - y_pred)) / max(np.sum(~np.isnan(y_true)),1e-7)
    return mae_value
import numpy as np

# Função para criar a matriz binária
# Função para criar a matriz binária com limite de corte variável
import numpy as np

# Função para criar a matriz binária com limite de corte variável
def create_binary_matrix(metrics_df, asmape_columns):
    # Inicializar a matriz binária com zeros
    binary_matrix = pd.DataFrame(0, index=metrics_df.index, columns=asmape_columns)
    
    # Para cada linha, definir um limite de corte baseado no mínimo da linha
    for i in range(len(metrics_df)):
        # Remover valores NaN da linha para calcular o mínimo
        non_nan_row = metrics_df[asmape_columns].iloc[i].dropna()
        
        if not non_nan_row.empty:
            # Calcular o limite de corte como 1.5 vezes o valor mínimo da linha
            min_value = non_nan_row.min()
            threshold = 1.5 * min_value if min_value > 1e-10 else 1.5 * 1e-10
            
            # Garantir que o threshold é um número finito
            if not np.isfinite(threshold):
                threshold = 1.5  # Valor padrão de fallback, ou ajuste conforme necessário

            # Preencher a matriz binária onde os valores são menores ou iguais ao limite de corte
            binary_matrix.iloc[i] = (metrics_df[asmape_columns].iloc[i] <= threshold).astype(int)
            
            # Se a linha da matriz binária tiver apenas zeros, ajustar para que o mínimo receba 1
            if binary_matrix.iloc[i].sum() == 0:
                min_index = non_nan_row.idxmin()
                binary_matrix.at[i, min_index] = 1

    return binary_matrix




# Suponho que `dirs` seja a lista de diretórios que você está percorrendo
for experiment in dirs:
    # Definir janelas dos experimentos
    base = experiment.split('/')
    wind = int(base[4])
    
    # Ler resultados
    real = pd.read_csv(experiment + '/real_full_data.csv', low_memory=False)
    whit_nan = pd.read_csv(experiment + '/input_full_data.csv', low_memory=False)
    inputeds = pd.read_csv(experiment + '/all_imputed.csv', low_memory=False)

    # DataFrame para armazenar os valores e métricas por janela
    windowed_data = []

    # Para cada série
    for s in np.unique(real['serie']):
        real_serie = real[real['serie'] == s].drop(columns=['serie'])  # Remover coluna da série
        nan_serie = whit_nan[whit_nan['serie'] == s].drop(columns=['serie'])  # Remover coluna da série
        imput_serie = inputeds[inputeds['serie'] == s].drop(columns=['serie'])  # Remover coluna da série

        # Para cada janela de dados dentro da série
        for i in range(0, real_serie.shape[0], wind):
            # Definir a janela nos dados reais, com NaN e imputados
            r_window = real_serie.iloc[i:i + wind]  # Dados reais
            nan_window = nan_serie.iloc[i:i + wind]  # Dados com NaN (máscara)
            imput_window = imput_serie.iloc[i:i + wind, :]  # Dados imputados

            # Criar máscara de NaNs - converter para um array booleano para aplicação em séries
            mask = nan_window.isna().to_numpy().flatten()

            # Lista para armazenar as janelas de valores com NaN, reais e métricas
            window_row = []

            # Adicionar o identificador da série 's' como a primeira coluna
            window_row.append(s)

            # 1. Adicionar os valores com NaN
            window_row.extend(nan_window.to_numpy().flatten())

            # 2. Adicionar os valores reais
            window_row.extend(r_window.to_numpy().flatten())

            # 3. Calcular as métricas (MAE e ASMAPE) para cada método e adicionar
            for method in imput_window.columns:
                imput_method_window = imput_window[method].to_numpy()

                # Selecionar apenas as posições onde havia NaN na entrada (usando a máscara)
                r_valid = r_window.to_numpy()[mask].flatten()
                imput_valid = imput_method_window[mask].flatten()

                # Remover os NaN antes de calcular as métricas
                valid_mask = ~np.isnan(r_valid) & ~np.isnan(imput_valid)
                r_valid_filtered = r_valid[valid_mask]
                imput_valid_filtered = imput_valid[valid_mask]

                # Verificar se há dados válidos (evitar cálculo em janelas completamente vazias)
                if len(r_valid_filtered) > 0:
                    # Calcular as métricas usando nansum
                    mae_value = mae_nansum(r_valid_filtered, imput_valid_filtered)
                    asmape_value = asmape(r_valid_filtered, imput_valid_filtered)
                else:
                    mae_value =500
                    asmape_value = 200 #máximo asmape

                # Adicionar as métricas para este método (MAE e ASMAPE)
                window_row.append(mae_value)
                window_row.append(asmape_value)

            # Armazenar os dados da janela
            windowed_data.append(window_row)

    # Criar DataFrame para armazenar os resultados
    # As primeiras colunas serão para os valores com NaN, depois para os reais, depois para as métricas
    mae_columns = [f"mae_{method}" for method in imput_window.columns]  # Colunas para MAE
    asmape_columns = [f"asmape_{method}" for method in imput_window.columns]  # Colunas para ASMAPE
    columns = ['serie'] + [f"val_nan_{i}" for i in range(wind)] + [f"val_real_{i}" for i in range(wind)] + mae_columns + asmape_columns
    multilabel_columns = [f"multabel_{method}" for method in imput_window.columns]
    
    # Criar DataFrame final
    metrics_df = pd.DataFrame(windowed_data, columns=columns)

    multiclass = np.argmin(metrics_df[asmape_columns],axis=1)
    worst_case = np.argmax(metrics_df[asmape_columns],axis=1)

    # Criar uma matriz binária onde 1 indica valores <= threshold e 0 indica valores > threshold

    binary_matrix = create_binary_matrix(metrics_df[asmape_columns], asmape_columns)


    metrics_df['multiclass'] = multiclass
    metrics_df['worst_case'] = worst_case
    metrics_df[multilabel_columns] = binary_matrix.to_numpy()
    ## Exibir resultados
    print(f"Métricas calculadas para o experimento {experiment}")


    # Salvar o DataFrame com as métricas calculadas 
    output_file = f"{experiment}/data_full_processed.csv"
    metrics_df.to_csv(output_file, index=False)
    # print(f"Salvou o arquivo de métricas em {output_file}")