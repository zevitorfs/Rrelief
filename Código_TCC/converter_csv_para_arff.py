import pandas as pd

# Carregar o CSV
df = pd.read_csv('GOCellcycleTreinamento_reduz_RrelifF.csv')

# Criar a estrutura ARFF
arff_header = "@RELATION dataset\n\n"

# Definir os atributos
arff_attributes = ""
for col in df.columns[:-1]:  # Para todas as colunas, exceto a última (classe)
    arff_attributes += f"@ATTRIBUTE {col} NUMERIC\n"
arff_attributes += f"@ATTRIBUTE classe {{{', '.join(df['classe'].unique())}}}\n\n"

# Definir os dados
arff_data = "@DATA\n"
for i, row in df.iterrows():
    arff_data += ','.join(map(str, row)) + "\n"

# Escrever o arquivo ARFF
with open("GOCellcycleTreinamento_reduz_RrelifF.arff", "w") as f:
    f.write(arff_header + arff_attributes + arff_data)
