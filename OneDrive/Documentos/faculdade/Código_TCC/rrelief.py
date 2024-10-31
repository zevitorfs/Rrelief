import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

# Função para calcular a diferença entre características
def diff(feature, instance1, instance2):
    return abs(instance1[feature] - instance2[feature])

# Função para calcular a diferença entre rótulos usando a distância de Hamming
def label_diff(labels1, labels2):
    set1 = set(labels1)
    set2 = set(labels2)
    return len(set1.symmetric_difference(set2))

# Função RReliefF
def rrelieff(X, y, k=5):
    m, f = X.shape
    Ndc = 0
    NdF = np.zeros(f)
    NdC_dF = np.zeros(f)
    W = np.zeros(f)
    
    for i in range(m):
        Ri = X[i]
        distances = pairwise_distances([Ri], X)[0]
        nearest_indices = np.argsort(distances)[1:k+1]
        
        for j in nearest_indices:
            Ij = X[j]
            d_ij = distances[j]
            Ndc += label_diff(y[i], y[j]) * d_ij
            
            for F in range(f):
                NdF[F] += diff(F, Ri, Ij) * d_ij
                NdC_dF[F] += label_diff(y[i], y[j]) * diff(F, Ri, Ij) * d_ij
    
    for F in range(f):
        W[F] = NdC_dF[F] / Ndc - (NdF[F] - NdC_dF[F]) / (m - Ndc)
    
    return W

# Carregar o dataset
df = pd.read_csv('GOCellcycleTeste.txt', header=None)

# Separar as características (features) dos rótulos (labels)
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].apply(lambda x: x.split('@')).values

# Exibir as primeiras linhas das características e dos rótulos
print("Características (primeiras 5 linhas):")
print(features[:5])
print("\nRótulos (primeiras 5 linhas):")
print(labels[:5])

# Executar o RReliefF com o dataset carregado
pesos = rrelieff(features, labels)
print("\nPesos das Características:")
print(pesos)