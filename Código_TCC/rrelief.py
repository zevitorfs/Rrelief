"""
Seleção de Atributos para Classificação Hierárquica Multirrótulo
HMC-RreliefF - versão adptada

@copright 2024
@author: José Victor Ferreira da Silva
IFMA

"""

import numpy as np

# Função para ler o arquivo de hierarquias
def ler_hierarquias(caminho_arquivo):
    hierarquia = {}
    with open(caminho_arquivo, 'r') as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if '/' in linha:
                pai, filho = linha.split('/')
                if filho not in hierarquia:
                    hierarquia[filho] = []
                hierarquia[filho].append(pai)
    return hierarquia

# Função para ler o dataset de atributos
def ler_dataset(caminho_arquivo):
    atributos = []
    classes = []
    with open(caminho_arquivo, 'r') as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            valores = linha.split(',')
            atributos.append([float(x) if x != '' else np.nan for x in valores[:-1]])
            classes.append(valores[-1].split('@'))
    return np.array(atributos), classes

# Função para calcular o caminho até a raiz
def caminho_ate_raiz(classe, hierarquia):
    caminho = set()
    fila = [classe]
    while fila:
        atual = fila.pop()
        if atual not in caminho:
            caminho.add(atual)
            fila.extend(hierarquia.get(atual, []))
    return caminho

# Função para calcular a distância hierárquica entre duas classes
def distancia_hierarquica(classe1, classe2, hierarquia):
    caminho1 = caminho_ate_raiz(classe1, hierarquia)
    caminho2 = caminho_ate_raiz(classe2, hierarquia)
    distancia = len(caminho1.symmetric_difference(caminho2))
    return distancia

# Função para calcular o peso dinâmico de uma classe em uma DAG
def calcula_peso(classe, hierarquia, w0):
    if classe not in hierarquia or not hierarquia[classe]:  # Caso base: raiz ou sem pais
        return w0

    pesos_pais = [calcula_peso(pai, hierarquia, w0) for pai in hierarquia[classe]]
    return w0 * (sum(pesos_pais) / len(pesos_pais))

# Função de diferença ponderada entre classes
def diff_classes(classes1, classes2, hierarquia, w0):
    soma = 0
    for c1 in classes1:
        for c2 in classes2:
            distancia = distancia_hierarquica(c1, c2, hierarquia)
            peso_c1 = calcula_peso(c1, hierarquia, w0)
            peso_c2 = calcula_peso(c2, hierarquia, w0)
            peso_medio = (peso_c1 + peso_c2) / 2
            soma += peso_medio * (distancia ** 2)
    return np.sqrt(soma)

# Função de diferença entre atributos
def diff_atributos(atributo1, atributo2):
    return abs(atributo1 - atributo2)  # Distância absoluta para atributos contínuos

# Implementação do HMC-RreliefF
def rrelieff(atributos, classes, hierarquia, k=15, w0=0.5):
    m, f = atributos.shape
    W = np.zeros(f)

    for i in range(m):
        Ri = atributos[i]
        Ci = classes[i]

        # Calcula distâncias e seleciona os k vizinhos mais próximos
        distancias = [np.linalg.norm(Ri - atributos[j]) for j in range(m) if j != i]
        vizinhos_indices = np.argsort(distancias)[:k]

        NdC = 0
        NdF = np.zeros(f)
        NdCF = np.zeros(f)

        for j in vizinhos_indices:
            Rj = atributos[j]
            Cj = classes[j]

            d_ij = np.exp(-np.linalg.norm(Ri - Rj))  # Peso baseado na proximidade no espaço dos atributos
            d_tau = diff_classes(Ci, Cj, hierarquia, w0)

            NdC += d_ij * d_tau
            for F in range(f):
                d_f = diff_atributos(Ri[F], Rj[F])
                NdF[F] += d_ij * d_f
                NdCF[F] += d_ij * d_tau * d_f

        for F in range(f):
            if NdC > 0 and (m - NdC) > 0:  # Evita divisões por zero
                W[F] += (NdCF[F] / NdC) - ((NdF[F] - NdCF[F]) / (m - NdC))

    return W

# Seleção de atributos com base nos pesos
def selecionar_atributos(atributos, pesos, num_atributos=None, limiar=None):
    if num_atributos:
        indices_selecionados = np.argsort(pesos)[-num_atributos:]
    elif limiar:
        indices_selecionados = np.where(pesos >= limiar)[0]
    else:
        raise ValueError("Forneça 'num_atributos' ou 'limiar'.")

    return atributos[:, indices_selecionados], indices_selecionados

# Função para gerar cabeçalho
def obter_cabecalho(atributos_indices):
    return [f"X{i+1}" for i in atributos_indices] + ["classe"]

# Função para gerar conjunto reduzido e salvar
def gera_conjuntos_reduzidos(atributos_reduzidos, classes, atributos_indices, nome_arquivo):
    cabecalho = obter_cabecalho(atributos_indices)
    with open(nome_arquivo, 'w') as arquivo:
        arquivo.write(",".join(cabecalho) + "\n")
        for i in range(len(atributos_reduzidos)):
            atributos_str = ",".join(map(str, atributos_reduzidos[i]))
            classes_str = "@".join(classes[i])
            arquivo.write(f"{atributos_str},{classes_str}\n")

#Caminhos para os arquivos
caminho_hierarquias = 'GoCellcycleRelacionamento.txt'
caminho_dataset = 'GOCellcycleTreinamento.txt'

#caminho_hierarquias = 'FicticioRelacionamento.txt'
#caminho_dataset = 'ficticio.txt'

# Leitura dos arquivos
hierarquias = ler_hierarquias(caminho_hierarquias)
atributos, classes = ler_dataset(caminho_dataset)

# Aplicação do RReliefF
pesos = rrelieff(atributos, classes, hierarquias, k=15, w0=0.5)

# Seleção dos melhores atributos
num_atributos_selecionados = 5
limiar = np.percentile(pesos, 75)
atributos_reduzidos, indices_selecionados = selecionar_atributos(atributos, pesos, limiar = limiar)

# Exibindo resultados
print("Pesos dos atributos:")
for i, peso in enumerate(pesos):
    print(f"Atributo {i + 1}: {peso:.4f}")

print("\nAtributos selecionados:")
print(f"Índices: {indices_selecionados + 1}")

# Salvando o novo dataset reduzido
gera_conjuntos_reduzidos(atributos_reduzidos, classes, indices_selecionados, 'GOCellcycleTreinamento_reduz_RrelifF.csv')
