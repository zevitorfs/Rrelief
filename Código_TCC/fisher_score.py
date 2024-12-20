"""
Seleção de Atributos para Classificação Hierárquica Multirrótulo
Fisher Score - versão 2.0

@copright 2022
@author: Raimundo Osvaldo Vieira
PPGCC - UTFPR Câmpus Ponta Grossa

"""
import pandas as pd
from operator import itemgetter

def retorna_ascendentes_por_classe(conjunto, classe, nivel):
    ascendentes_classe = {}
    conjunto_f = conjunto[conjunto['filho'].str.contains(classe)]
    ascendentes_classe[classe] = nivel

    for index, row in conjunto_f.iterrows():
        if row['pai'] != 'RAIZ':
            retorno = retorna_ascendentes_por_classe(conjunto, row['pai'], ascendentes_classe[classe]+1)
            retorno.update(ascendentes_classe)
            ascendentes_classe = retorno
        
    return ascendentes_classe

def retorna_ascendentes(classes, hierarquia):
    print("... Descobrindo Ascendentes ...\n")
    ascendentes = {}
    
    for ck in classes:
        ascendentes[ck] = retorna_ascendentes_por_classe(hierarquia, ck, 0)

    return ascendentes    

def maior_nivel(ascendentes):
    print("... Descobrindo Nivel Maximo ...")
    maior_nivel = 0
    
    for tipo, tipo2 in ascendentes.items():
        maior = tipo2[max(tipo2, key=tipo2.get)]
        if maior > maior_nivel:
            maior_nivel = maior
    
    return maior_nivel

def calcula_fisher_score(conjunto, hierarquia):
    fisher = []
    atributos = conjunto.iloc[:,:-1]
    classes = set()
    ascendentes = {}
    
    for c in conjunto.iloc[:,-1]:
        rotulos = c.split('@')
        for ck in rotulos:
            classes.add(ck)
    
    ascendentes = retorna_ascendentes(classes, hierarquia) 
    max_nivel = maior_nivel(ascendentes)
    print(max_nivel)
    
    for coluna, item in atributos.iteritems():
    
        #print("Atributo: ", coluna)
        media = item.mean()
        print("Media do atributo: ", coluna, ": ", media)
        soma_stk = 0.0
        media_k = 0.0
        
        for ck in classes:
            
            #print("CLASSE ...\n")
            
            soma_valores = 0.0
            soma_pesos = 0.0
            Ek = pd.DataFrame()
            
            dados = conjunto
            
            ascendentes_c = ascendentes[ck] 
            
            for ch, nivel in sorted(ascendentes_c.items(), key=itemgetter(1)):
                
                #print("...\n")
                
                I = dados[dados.iloc[:,-1].str.contains(ch)]
                dados = dados.drop(I.index)
                
                if I.empty == False:
                    Ek = pd.concat([Ek, I], axis=0)
                    I = I.iloc[:,I.columns.get_loc(coluna)]
                    peso = 1 - ascendentes_c[ch]/(max_nivel+1)
                    soma_valores += I.sum() * peso
                    soma_pesos += I.count() * peso
                
            media_ponderada = soma_valores/soma_pesos  
            nk = Ek.shape[0]
            
            stk = 0
            for xij in Ek.iloc[:,Ek.columns.get_loc(coluna)]:
                stk += (xij - media_ponderada)**2
            
            media_k += nk*((media_ponderada-media)**2)
            soma_stk += stk     
            
        fisher.append(media_k/soma_stk)
        
    print(fisher)
      
    return pd.DataFrame(fisher)

def seleciona_atributos(conjunto, hierarquia):
    fisher_score = calcula_fisher_score(conjunto, hierarquia)
    print(fisher_score[0].mean())
    return fisher_score[fisher_score[0] > fisher_score[0].mean()].index.tolist()

def obter_cabecalho(indices):
    rotulos = []
    for rotulo in indices:
        rotulos.append("X"+str(rotulo))
    return rotulos
    
def gera_conjuntos_reduzidos(conjunto_treino, conjunto_teste, atributos, nome_arquivo):
    conjunto_reduzido_teste = pd.concat([conjunto_teste.iloc[:,atributos],conjunto_teste.iloc[:,-1]],axis=1)
    id_arquivo = nome_arquivo+"Teste_reduz.csv"
    conjunto_reduzido_teste.to_csv('datasets/BaseFicticia/'+id_arquivo, header=obter_cabecalho(conjunto_reduzido_teste.head()), index=False)
    id_arquivo = nome_arquivo+"Treinamento_reduz.csv"
    conjunto_reduzido_treino = pd.concat([conjunto_treino.iloc[:,atributos],conjunto_treino.iloc[:,-1]],axis=1)
    conjunto_reduzido_treino.to_csv('datasets/BaseFicticia/'+id_arquivo, header=obter_cabecalho(conjunto_reduzido_treino.head()), index=False)

#arquivos = ['GOCellcycle', 'GoChurch', 'GODerisi', 'GoEisen', 'GOExpr', 'GOGasch1', 'GOGasch2', 'GOPheno', 'GOSeq', 'GOSpo']
arquivos = ['Ficticio']


for arquivo in arquivos:
    id_arquivo = arquivo+"Relacionamento.txt"
    df_hierarquia = pd.read_csv('datasets/BaseFicticia/'+id_arquivo, names=['pai', 'filho'], sep="/")
    id_arquivo = arquivo+"Teste.txt"
    df_teste = pd.read_csv('datasets/BaseFicticia/'+id_arquivo, header = None)
    id_arquivo = arquivo+"Treinamento.txt"
    df_treino = pd.read_csv('datasets/BaseFicticia/'+id_arquivo, header = None) 
    gera_conjuntos_reduzidos(df_treino, df_teste, seleciona_atributos(df_treino, df_hierarquia),arquivo)