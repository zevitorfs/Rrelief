import numpy as np


def calcular_importancia_rrelief(dados, resultados, num_vizinhos):
    """
    Calcula a importância de cada característica usando o algoritmo RReliefF adaptado para classificação hierárquica multirrótulo.

    Parâmetros:
    - dados: matriz de dados (numpy array ou lista de listas)
    - resultados: lista de listas de rótulos (multirrótulo)
    - num_vizinhos: número de vizinhos mais próximos a considerar

    Retorna:
    - Importância de cada característica (lista)
    """
    num_instancias, num_caracteristicas = dados.shape
    Ndc = 0
    NdF = np.zeros(num_caracteristicas)
    NdC_dF = np.zeros(num_caracteristicas)
    W = np.zeros(num_caracteristicas)

    for i in range(num_instancias):
        instancia_atual = dados[i]
        distancias = np.linalg.norm(dados - instancia_atual, axis=1)
        vizinhos_proximos = np.argsort(distancias)[1:num_vizinhos + 1]

        for j in vizinhos_proximos:
            vizinho = dados[j]
            peso = np.exp(-distancias[j])

            # Atualiza Ndc com a diferença ponderada entre os rótulos
            Ndc += calcular_diferenca_rotulos(resultados[i], resultados[j]) * peso

            for F in range(num_caracteristicas):
                diferenca_caracteristica = abs(dados[i][F] - dados[j][F])

                # Atualiza NdF[F] com a diferença ponderada da característica
                NdF[F] += diferenca_caracteristica * peso

                # Atualiza NdC&dF[F] com a diferença ponderada entre os rótulos e a característica
                NdC_dF[F] += calcular_diferenca_rotulos(resultados[i], resultados[j]) * diferenca_caracteristica * peso

    for F in range(num_caracteristicas):
        if Ndc > 0:  # Para evitar divisão por zero
            W[F] = NdC_dF[F] / Ndc - (NdF[F] - NdC_dF[F]) / (num_instancias - Ndc)

    return W


def calcular_diferenca_rotulos(rotulos1, rotulos2):
    """
    Calcula a diferença entre dois conjuntos de rótulos.

    Parâmetros:
    - rotulos1: lista de rótulos da primeira instância
    - rotulos2: lista de rótulos da segunda instância

    Retorna:
    - Diferença entre os rótulos (float)
    """
    return sum(1 for r1, r2 in zip(rotulos1, rotulos2) if r1 != r2)


# Função para calcular a distância euclidiana entre duas instâncias (linhas de dados)
def distancia_euclidiana(linha1, linha2):
    return np.linalg.norm(np.array(linha1) - np.array(linha2))


# Exemplo de uso
dados = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
resultados = [[0, 1], [1, 2], [0, 2], [1, 3]]
num_vizinhos = 2

importancia = calcular_importancia_rrelief(dados, resultados, num_vizinhos)
print(importancia)