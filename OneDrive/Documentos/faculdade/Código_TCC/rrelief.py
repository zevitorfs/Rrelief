import numpy as np
import os

def load_txt_dataset(file_path, label_columns=2, delimiter=","):
    """
    Carrega um dataset no formato .txt.

    Espera que o arquivo tenha:
    - As primeiras colunas como características (valores numéricos).
    - As últimas colunas como rótulos binários.

    Parâmetros:
    - file_path: Caminho do arquivo .txt.
    - label_columns: Número de colunas de rótulos binários.
    - delimiter: Delimitador usado no arquivo .txt.

    Retorna:
    - X: Matriz de características (m x n).
    - y: Matriz de rótulos binários (m x p).
    """
    try:
        data = np.loadtxt(file_path, delimiter=delimiter)
        if data.shape[1] <= label_columns:
            raise ValueError("Número de colunas de rótulos é maior ou igual ao número total de colunas no dataset.")
        
        X = data[:, :-label_columns]
        y = data[:, -label_columns:]
        return X, y
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return None, None

def diff(S_i, S_j, weights):
    diff_squared = weights * (S_i - S_j) ** 2
    return np.sqrt(np.sum(diff_squared))

def RReliefF(X, y, k, weights):
    """
    Implementação do algoritmo RReliefF adaptado para Python
    
    Parâmetros:
    X : np.ndarray
        Matriz de características.
    y : np.ndarray
        Vetor de rótulos.
    k : int
        Número de vizinhos mais próximos a considerar.
    weights : np.ndarray
        Pesos para as características.
    
    Retorna:
    np.ndarray
        Vetor de pesos das características.
    """
    m, n = X.shape
    _, p = y.shape

    N_DC = np.zeros(n)
    N_DF = np.zeros(n)
    N_DC_and_DF = np.zeros(n)
    W = np.zeros(n)

    # Pré-calcular todas as distâncias
    all_distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)

    for i in range(m):
        R_i = X[i]
        S_i = y[i]

        distances = all_distances[i]
        nearest_indices = np.argsort(distances)[: k + 1]
        nearest_indices = nearest_indices[nearest_indices != i]

        for j in nearest_indices:
            R_j = X[j]
            S_j = y[j]

            diff_target = diff(S_i, S_j, weights)

            for f in range(n):
                diff_feature = abs(R_i[f] - R_j[f])
                N_DC[f] += diff_target * distances[j]
                N_DF[f] += diff_feature * distances[j]
                N_DC_and_DF[f] += diff_target * diff_feature * distances[j]

    for f in range(n):
        if N_DF[f] != 0:
            W[f] = N_DC_and_DF[f] / N_DF[f] - N_DC[f] / N_DF[f]

    return W

# Carregar o dataset
file_path = "C:/Users/josev/OneDrive/Documentos/faculdade/Rrelief/OneDrive/Documentos/faculdade/Código_TCC/GOCellcycleTeste.txt"

# Verificar se o arquivo existe
if not os.path.isfile(file_path):
    print(f"Arquivo {file_path} não encontrado.")
else:
    X, y = load_txt_dataset(file_path, label_columns=2, delimiter=",")

    if X is not None and y is not None:
        # Pesos dos rótulos
        weights = [0.5, 0.5]

        # Número de vizinhos
        k = 2

        # Executa o algoritmo
        W = RReliefF(X, y, k, weights)
        print(f"Relevância das características: {W}")

        # Salvar os resultados em um arquivo .txt
        np.savetxt("melhores_caracteristicas.txt", W, delimiter=",")
    else:
        print("Falha ao carregar o dataset.")
