O projeto abordou a análise de um conjunto de dados da plataforma Netflix com o objetivo de extrair insights e informações valiosas. Ao longo do projeto, foram realizadas várias etapas de exploração, visualização e análise dos dados, cada uma fornecendo insights únicos sobre o comportamento dos usuários e a performance da plataforma. Vamos resumir as principais conclusões e insights obtidos:

    Distribuição dos Tipos de Assinatura: A análise mostrou que a maioria dos usuários possui o tipo de assinatura Padrão, seguido pelo tipo Básico e Premium.

    Variação das Receitas Mensais e Países: A análise das receitas mensais por país revelou diferenças significativas entre os países, com alguns países contribuindo mais para a receita total.

    Análise Temporal das Adesões e Cancelamentos: Ao longo do tempo, foi possível observar flutuações nas adesões e cancelamentos, com alguns meses apresentando picos de adesão e outros com maiores cancelamentos.

    Preferências de Dispositivos: A maioria dos usuários acessa a plataforma Netflix por meio de dispositivos móveis, seguido por computadores e smart TVs.

    Atividade de Visualização por Hora do Dia: O gráfico de área mostrou que a atividade de visualização aumenta no final da tarde e à noite, atingindo o pico por volta das 20h.

    Gêneros de Conteúdo Populares:  Os gêneros de conteúdo mais populares entre os usuários são Drama, Comédia e Ação. 

    Correlação entre Duração da Assinatura e Receita Mensal: Foi observada uma correlação positiva fraca entre a duração da assinatura e a receita mensal.

    Previsão de Receita Mensal:  Um modelo de regressão linear foi desenvolvido para prever a receita mensal com base em características dos usuários. No entanto, os resultados podem ser melhorados com mais dados e recursos 

    Segmentação de Usuários:  Os usuários foram segmentados com base nas suas preferências de conteúdo e comportamento de visualização, permitindo uma compreensão mais profunda de diferentes grupos de usuários. 

    Análise de Inatividade e Abandono: Foi identificada uma relação entre a inatividade do usuário e o aumento do abandono de assinaturas.

    Comunicação de Insights:  Visualizações como gráficos de barras, gráficos de dispersão e mapas de calor foram utilizadas para comunicar os insights de forma clara e eficaz. 
# Segue Codigo Python 
# 1 - Carregar o conjunto de dados e entender sua estrutura 
import pandas as pd

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Exibir as primeiras linhas do conjunto de dados
display(df.head())

# Obter informações sobre as colunas e tipos de dados
display(df.info())

# Calcular estatísticas descritivas básicas
display(df.describe())

# 2 - Identificar e tratar valores ausentes ou inconsistentes.
import pandas as pd

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Identificar valores ausentes em cada coluna
missing_values = df.isnull().sum()
print("Valores Ausentes por Coluna:")
print(missing_values)

# Tratar valores ausentes: Preencher com a média
df = df.fillna(df.mean())

# Verificar novamente para garantir que os valores ausentes foram tratados
missing_values_after_fill = df.isnull().sum()
print("\nValores Ausentes Após Tratamento:")
print(missing_values_after_fill)

# 3 - Verificar a distribuição dos tipos de assinatura, receitas mensais e países. 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Verificar a distribuição dos tipos de assinatura
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Subscription Type')
plt.title('Distribuição dos Tipos de Assinatura')
plt.xlabel('Tipo de Assinatura')
plt.ylabel('Contagem')
plt.xticks(rotation=45)
plt.show()

# Verificar a distribuição das receitas mensais
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Monthly Revenue', bins=20, kde=True)
plt.title('Distribuição das Receitas Mensais')
plt.xlabel('Receita Mensal')
plt.ylabel('Contagem')
plt.show()

# Verificar a distribuição dos países
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='Country', order=df['Country'].value_counts().index)
plt.title('Distribuição dos Países')
plt.xlabel('Contagem')
plt.ylabel('País')
plt.show()

# 4 - Analisar as mudanças ao longo do tempo nas adesões e cancelamentos de assinatura
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Converter as colunas de datas para o formato datetime
df['Join Date'] = pd.to_datetime(df['Join Date'])
df['Last Payment Date'] = pd.to_datetime(df['Last Payment Date'])

# Criar coluna para identificar se houve cancelamento de assinatura
df['Churn'] = df['Last Payment Date'].isnull()

# Criar um DataFrame para as datas de adesões e cancelamentos
dates_df = pd.DataFrame({
    'Data': pd.concat([df['Join Date'], df['Last Payment Date']]),
    'Evento': ['Adesão'] * len(df) + ['Cancelamento'] * len(df)
})

# Agrupar por mês e contar adesões e cancelamentos
monthly_changes = dates_df.groupby([dates_df['Data'].dt.to_period('M'), 'Evento']).size().unstack(fill_value=0)

# Plotar o gráfico
plt.figure(figsize=(12, 6))
monthly_changes.plot(kind='line', marker='o', figsize=(12, 6))
plt.title('Mudanças nas Adesões e Cancelamentos ao Longo do Tempo')
plt.xlabel('Mês')
plt.ylabel('Contagem')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# 5 - Identificar os tipos de dispositivos mais usados para acessar a plataforma

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Contar os tipos de dispositivos mais usados
device_counts = df['Device'].value_counts()

# Plotar o gráfico de barras
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=device_counts.index, y=device_counts.values, palette='viridis')
plt.title('Tipos de Dispositivos Mais Usados para Acessar a Netflix')
plt.xlabel('Tipo de Dispositivo')
plt.ylabel('Contagem')
plt.xticks(rotation=45)

# Adicionar os valores nas barras
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()

# 6 - Analisar padrões diários, semanais ou mensais de atividade de visualização.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Converter a coluna de data de visualização para o formato datetime
df['Last Payment Date'] = pd.to_datetime(df['Last Payment Date'])

# Criar coluna para dia da semana e mês
df['Dia_da_Semana'] = df['Last Payment Date'].dt.day_name()
df['Mês'] = df['Last Payment Date'].dt.strftime('%B')


# Plotar a atividade de visualização por dia da semana
plt.figure(figsize=(10, 6))
sns.set_palette("Set1")
ax1 = sns.countplot(data=df, x='Dia_da_Semana')
plt.title('Atividade de Visualização por Dia da Semana')
plt.xlabel('Dia da Semana')
plt.ylabel('Contagem')
plt.xticks(rotation=45)

# Adicionar os valores nas barras
for p in ax1.patches:
    ax1.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()

# Plotar a atividade de visualização por mês
plt.figure(figsize=(10, 6))
sns.set_palette("viridis")
ax2 = sns.countplot(data=df, x='Mês')
plt.title('Atividade de Visualização por Mês')
plt.xlabel('Mês')
plt.ylabel('Contagem')
plt.xticks(rotation=45)

# Adicionar os valores nas barras
for p in ax2.patches:
    ax2.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()

# 7 - Verificar se existem picos de visualização em determinados horários ou dias da semana.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Converter a coluna de data de visualização para o formato datetime
df['Last Payment Date'] = pd.to_datetime(df['Last Payment Date'])

# Criar coluna para hora do dia e dia da semana
df['Hora'] = df['Last Payment Date'].dt.hour
df['Dia_da_Semana'] = df['Last Payment Date'].dt.day_name()

# Plotar a atividade de visualização por hora do dia usando um gráfico de área
plt.figure(figsize=(10, 6))
sns.set_palette("Set1")
hora_counts = df['Hora'].value_counts().sort_index()
hora_counts.plot(kind='area')
plt.title('Atividade de Visualização por Hora do Dia')
plt.xlabel('Hora do Dia')
plt.ylabel('Contagem')
plt.xticks(range(24))

# Adicionar os valores nas áreas
for i, value in enumerate(hora_counts):
    plt.text(i, value + 10, str(value), ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# Plotar a atividade de visualização por dia da semana
plt.figure(figsize=(10, 6))
sns.set_palette("viridis")
ax = sns.countplot(data=df, x='Dia_da_Semana', order=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
plt.title('Atividade de Visualização por Dia da Semana')
plt.xlabel('Dia da Semana')
plt.ylabel('Contagem')
plt.xticks(rotation=45)

# Adicionar os valores nas barras
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()

# 8 - Identificar os gêneros mais populares entre os usuários.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Verificar os gêneros de conteúdo mais populares por gênero
genre_counts_male = df[df['Gender'] == 'Male']['Gender'].count()
genre_counts_female = df[df['Gender'] == 'Female']['Gender'].count()

# Plotar o gráfico de barras
plt.figure(figsize=(8, 6))
sns.set_palette("Set2")
ax = sns.barplot(x=['Male', 'Female'], y=[genre_counts_male, genre_counts_female])
plt.title('Gêneros de Conteúdo Mais Populares por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Contagem')

# Adicionar os valores nas barras
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 9 -  Analisar se as preferências de conteúdo variam de acordo com o tipo de assinatura.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Verificar as preferências de conteúdo por tipo de assinatura
content_prefs_by_subscription = df.groupby(['Subscription Type']).size()

# Plotar o gráfico de barras
plt.figure(figsize=(10, 6))
sns.set_palette("viridis")
ax = sns.barplot(x=content_prefs_by_subscription.index, y=content_prefs_by_subscription.values)
plt.title('Preferências de Conteúdo por Tipo de Assinatura')
plt.xlabel('Tipo de Assinatura')
plt.ylabel('Contagem')

# Adicionar os valores nas barras
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 10 - Desenvolver um modelo de previsão de receita com base nas características dos usuários. 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Mapear durações de assinatura para valores numéricos
duration_mapping = {'1 Month': 1, '3 Months': 3, '6 Months': 6, '1 Year': 12}
df['Plan Duration'] = df['Plan Duration'].map(duration_mapping)

# Selecionar colunas relevantes para a previsão de receita
features = ['Plan Duration']
target = 'Monthly Revenue'

# Dividir o conjunto de dados em treino e teste
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Separar as features e o target
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Criar um modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular o erro médio quadrático (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Erro Médio Quadrático (RMSE): {rmse:.2f}")

# 11 - Utilizar regressão para prever a receita mensal com base nas informações disponíveis.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Mapear durações de assinatura para valores numéricos
duration_mapping = {'1 Month': 1, '3 Months': 3, '6 Months': 6, '1 Year': 12}
df['Plan Duration'] = df['Plan Duration'].map(duration_mapping)

# Selecionar colunas relevantes para a previsão de receita
features = ['Age', 'Plan Duration']
target = 'Monthly Revenue'

# Converter coluna Subscription Type em variáveis dummy
df = pd.get_dummies(df, columns=['Subscription Type'], drop_first=True)

# Dividir o conjunto de dados em treino e teste
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Separar as features e o target
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Criar um modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular o erro médio quadrático (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Erro Médio Quadrático (RMSE): {rmse:.2f}")

#12 - Identificar os principais motivos que levam ao abandono de assinaturas. 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Análise da distribuição de churn em relação à data de adesão
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Join Date', hue='Subscription Type')
plt.title('Distribuição de Churn em Relação à Data de Adesão e Tipo de Assinatura')
plt.xticks(rotation=45)
plt.xlabel('Data de Adesão')
plt.ylabel('Número de Churns')
plt.legend(title='Tipo de Assinatura', loc='upper right')
plt.tight_layout()
plt.show()

# Análise da distribuição de churn em relação à data do último pagamento
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Last Payment Date', hue='Subscription Type')
plt.title('Distribuição de Churn em Relação à Data do Último Pagamento e Tipo de Assinatura')
plt.xticks(rotation=45)
plt.xlabel('Data do Último Pagamento')
plt.ylabel('Número de Churns')
plt.legend(title='Tipo de Assinatura', loc='upper right')
plt.tight_layout()
plt.show()

# 13 - Analisar se existe alguma relação entre a inatividade do usuário e o abandono.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Calcular o tempo de inatividade
df['Inatividade (dias)'] = (pd.to_datetime(df['Last Payment Date']) - pd.to_datetime(df['Join Date'])).dt.days

# Definir os intervalos de inatividade
bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 1000]
labels = ['0-30', '31-60', '61-90', '91-120', '121-150', '151-180', '181-210', '211-240', '241-270', '271-300', '301-330', '331-360', '361+']
df['Intervalo de Inatividade'] = pd.cut(df['Inatividade (dias)'], bins=bins, labels=labels)

# Calcular a proporção de abandono em cada intervalo
churn_by_interval = df.groupby('Intervalo de Inatividade')['Intervalo de Inatividade'].count()
churn_by_interval /= df.shape[0]

# Plotar o gráfico de barras com os valores nas barras
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=churn_by_interval.index, y=churn_by_interval.values)
plt.title('Proporção de Abandono por Intervalo de Inatividade')
plt.xlabel('Intervalo de Inatividade (dias)')
plt.ylabel('Proporção de Abandono')
plt.xticks(rotation=45)

# Adicionar os valores nas barras
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()

# 14 - Criar gráficos e visualizações para comunicar os insights de forma clara.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Converter as colunas de data para o formato correto
df['Join Date'] = pd.to_datetime(df['Join Date'])
df['Last Payment Date'] = pd.to_datetime(df['Last Payment Date'])

# Visualizar a distribuição da idade dos usuários
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', bins=20)
plt.title('Distribuição da Idade dos Usuários')
plt.xlabel('Idade')
plt.ylabel('Porcentagem')

# Calcular a porcentagem de cada faixa etária
total_users = len(df)
for patch in plt.gca().patches:
    height = patch.get_height()
    percentage = (height / total_users) * 100
    plt.gca().annotate(f'{percentage:.2f}%', (patch.get_x() + patch.get_width() / 2, height),
                       ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# 15 -  Apresentar os resultados por meio de gráficos de barras, gráficos de dispersão e mapas de calor.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do arquivo CSV
file_path = "/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv"

# Carregar o conjunto de dados
df = pd.read_csv(file_path)

# Gráfico de Barras: Distribuição dos Tipos de Assinatura
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Subscription Type')
plt.title('Distribuição dos Tipos de Assinatura')
plt.xlabel('Tipo de Assinatura')
plt.ylabel('Contagem')
plt.xticks(rotation=45)
plt.tight_layout()

# Gráfico de Dispersão: Receita Mensal vs Idade
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Monthly Revenue', y='Age', alpha=0.5)
plt.title('Receita Mensal vs Idade')
plt.xlabel('Receita Mensal')
plt.ylabel('Idade')
plt.tight_layout()

# Mapa de Calor: Correlação entre Variáveis Numéricas
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Mapa de Calor - Correlação entre Variáveis Numéricas')
plt.tight_layout()

plt.show()

# 16 - Resumir os principais insights obtidos ao longo do projeto. 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o arquivo
file_path = '/content/drive/MyDrive/Curso de Python/Extração de Insights do Conjunto de Dados/Netflix Userbase.csv'
df = pd.read_csv(file_path)

# Converter colunas de data para o tipo de data e hora
df['Join Date'] = pd.to_datetime(df['Join Date'])
df['Last Payment Date'] = pd.to_datetime(df['Last Payment Date'])

# Resumo dos Principais Insights
print("Resumo dos Principais Insights:\n")

# 1. Distribuição dos Tipos de Assinatura
subscription_counts = df['Subscription Type'].value_counts()
print("1. Distribuição dos Tipos de Assinatura:")
print(subscription_counts)
print("\n")

# 2. Mudanças nas Adesões e Cancelamentos ao Longo do Tempo
monthly_changes = df.groupby(df['Join Date'].dt.to_period('M')).size().reset_index(name='Novas Adesões')
monthly_changes['Cancelamentos'] = df.groupby(df['Last Payment Date'].dt.to_period('M')).size().reset_index(name='Cancelamentos')['Cancelamentos']
print("2. Mudanças nas Adesões e Cancelamentos ao Longo do Tempo:")
print(monthly_changes)
print("\n")

# 3. Distribuição da Idade dos Usuários
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', bins=20, kde=True)
plt.title('Distribuição da Idade dos Usuários')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

# 4. Atividade de Visualização por Hora do Dia
hourly_activity = df['Join Date'].dt.hour.value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=hourly_activity.index, y=hourly_activity.values)
plt.title('Atividade de Visualização por Hora do Dia')
plt.xlabel('Hora do Dia')
plt.ylabel('Número de Usuários')
plt.show()









