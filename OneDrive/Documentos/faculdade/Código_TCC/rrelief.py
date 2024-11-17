import numpy as np

def load_txt_dataset(file_path, label_columns=2):
    """
    Carrega um dataset no formato .txt.

    Espera que o arquivo tenha:
    - As primeiras colunas como características (valores numéricos).
    - As últimas colunas como rótulos binários.

    Parâmetros:
    - file_path: Caminho do arquivo .txt.
    - label_columns: Número de colunas de rótulos binários.

    Retorna:
    - X: Matriz de características (m x n).
    - y: Matriz de rótulos binários (m x p).
    """
    try:
        data = np.loadtxt(file_path, delimiter=',')  # Carrega os dados (espera que sejam separados por vírgula)
        X = data[:, :-label_columns]  # Todas as colunas menos as últimas (assumindo que são rótulos)
        y = data[:, -label_columns:]  # Últimas colunas como rótulos binários
        return X, y
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return None, None

def diff(S_i: np.ndarray, S_j: np.ndarray, weights: np.ndarray) -> float:
    """
    Calcula a diferença no espaço-alvo entre duas instâncias com base nas descrições binárias dos rótulos.

    Parâmetros:
    - S_i: Vetor de rótulos binários da instância i.
    - S_j: Vetor de rótulos binários da instância j.
    - weights: Vetor de pesos para cada rótulo.

    Retorna:
    - A diferença no espaço-alvo ponderada pelos pesos.
    """
    diff_squared = weights * (S_i - S_j)**2
    return np.sqrt(np.sum(diff_squared))

def RReliefF(X, y, k, weights):
    """
    Implementação do algoritmo RReliefF adaptado para Python.
    """
    m, n = X.shape
    _, p = y.shape

    N_DC = np.zeros(n)
    N_DF = np.zeros(n)
    N_DC_and_DF = np.zeros(n)
    W = np.zeros(n)

    for i in range(m):
        R_i = X[i]
        S_i = y[i]

        distances = np.linalg.norm(X - R_i, axis=1)
        nearest_indices = np.argsort(distances)[:k+1]
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
        if N_DC[f] != 0:
            W[f] = (N_DC_and_DF[f] / N_DC[f]) - ((N_DF[f] - N_DC_and_DF[f]) / (m - N_DC[f]))

    return W

# Caminho do arquivo .txt
file_path = "GOCellcycleTeste.txt"

# Carrega o dataset
X, y = load_txt_dataset(file_path)

if X is not None and y is not None:
    # Pesos dos rótulos
    weights = [0.5, 0.5]

    # Número de vizinhos
    k = 2

    # Executa o algoritmo
    W = RReliefF(X, y, k, weights)
    print(f"Relevância das características: {W}")
else:
    print("Falha ao carregar o dataset.")