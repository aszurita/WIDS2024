import pandas as pd
training_df  = pd.read_csv("assets/data/training.csv")
corre_df=pd.read_csv("assets/data/correlaciones.csv")

def round_value(val):
    return round(val, 6)

def dicc_correla(training_df):
    Numerical_Columns = [column for column in training_df.columns if training_df[column].dtype != object ]
    df_numericalColumns = training_df[Numerical_Columns]
    df_corr = df_numericalColumns.corr()
    umbral = 0.7
    dic_corre = {}
    num_columnas = len(df_corr.columns)
    for i in range(num_columnas):
        for j in range(i + 1, num_columnas):
            correlacion = df_corr.iloc[i, j]
            if abs(correlacion) > umbral:
                columna_i = df_corr.columns[i]
                columna_j = df_corr.columns[j]
                if columna_i not in dic_corre:
                    dic_corre[columna_i] = []
                dic_corre[columna_i].append((columna_j, correlacion))
                dic_corre[columna_i]=sorted(dic_corre[columna_i], key=lambda x: abs(x[1]), reverse=True)
    df = pd.DataFrame([(k, *t) for k, v in dic_corre.items() for t in v], columns=['source', 'target', 'value'])#como los grafos
    df['value'] = df['value'].apply(round_value)
    file_path = 'df_corre.csv'
    df.to_csv(file_path, index=False)

dicc_correla(training_df)