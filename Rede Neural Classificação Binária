import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt

# Download de ambas as bases de dados
url_treinamento = "https://www.dropbox.com/scl/fi/e0h7mi560638dezl49tlz/dados_pessoas_treinamento_RN.csv?rlkey=z3ekk9f14yf6s5q52glogldsl&st=v7yz8j4d&dl=1"
url_teste = "https://www.dropbox.com/scl/fi/frbtxibd6pakghucc9jqx/dados_pessoas_teste_RN.csv?rlkey=cor2qtgu0rbyzd7z95bursgdv&st=jnd0azal&dl=1"
urllib.request.urlretrieve(url_treinamento, "dados_pessoas_treinamento_RN.csv")
urllib.request.urlretrieve(url_teste, "dados_pessoas_teste_RN.csv")

# Carregando e convertendo os dados
df_treino = pd.read_csv("dados_pessoas_treinamento_RN.csv", header=None)
df_teste = pd.read_csv("dados_pessoas_teste_RN.csv", header=None)

X = torch.tensor(df_treino.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(df_treino.iloc[:, -1].values.reshape(-1, 1), dtype=torch.float32)
X_test = torch.tensor(df_teste.iloc[:, :-1].values, dtype=torch.float32)
y_test = torch.tensor(df_teste.iloc[:, -1].values.reshape(-1, 1), dtype=torch.float32)

# Função para dividir o banco de dado de treinamento em dado de treino e dado de validação
# optou-se por manter uma proporção de 0.8 e 0.2
def dividir_treino_validacao(X, y, proporcao_treino=0.8):
    dataset = TensorDataset(X, y)
    tamanho_treino = int(proporcao_treino * len(dataset))
    tamanho_validacao = len(dataset) - tamanho_treino
    return random_split(dataset, [tamanho_treino, tamanho_validacao])

# Implementação do modelo de rede neural, após testes foi mantido uma camada oculta com 50 neuronios
class ClassificadorBinario(nn.Module):
    def __init__(self, input_size):
        super(ClassificadorBinario, self).__init__()
        self.hidden = nn.Linear(input_size, 50)
        self.output = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

# Preparando os dados para uso, optou-se por usar um batch size de 32
# diante dos resultados optidos com tal valor apos testes
dataset_treino, dataset_validacao = dividir_treino_validacao(X, y)
train_loader = DataLoader(dataset_treino, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset_validacao, batch_size=32)

input_size = X.shape[1]
model = ClassificadorBinario(input_size)

# Criação dos Hiperparâmetros
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
epochs = 1500

# Criação bvetores para para armazenar os valores de perda entre treino
# e validação
losses_treino = []
losses_validacao = []

# Realização do Treinamento com validação por epoca
for epoca in range(epochs):
    model.train()
    loss_treino = 0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_treino += loss.item()


    model.eval()
    loss_validacao = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            y_val_pred = model(X_val)
            loss = loss_fn(y_val_pred, y_val)
            loss_validacao += loss.item()

    losses_treino.append(loss_treino)
    losses_validacao.append(loss_validacao)
    print(f'Época {epoca+1} | Loss Treino: {loss_treino:.4f} | Loss Validação: {loss_validacao:.4f}')

# Gráfico de loss em relação a epoca
plt.figure(figsize=(10, 5))
plt.plot(losses_treino, label="Loss Treino")
plt.plot(losses_validacao, label="Loss Validação")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Loss por Época")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Avaliação completa do conjunto de treinamento

print("\nAvaliação no conjunto de treinamento completo:")
model.eval()
with torch.no_grad():
    y_pred_train = model(X)
    y_pred_train_labels = (y_pred_train >= 0.5).float()
    acuracia_treino = (y_pred_train_labels == y).float().mean()
    print(f"Acurácia no conjunto de treinamento: {acuracia_treino:.4f}")

    tp_train = ((y_pred_train_labels == 1) & (y == 1)).sum()
    tn_train = ((y_pred_train_labels == 0) & (y == 0)).sum()
    fp_train = ((y_pred_train_labels == 1) & (y == 0)).sum()
    fn_train = ((y_pred_train_labels == 0) & (y == 1)).sum()

    print("\nMatriz de Confusão (Treinamento):")
    print(f"                Previsto 0    Previsto 1")
    print(f"Real 0      {tn_train.item():>10}    {fp_train.item():>10}")
    print(f"Real 1      {fn_train.item():>10}    {tp_train.item():>10}")

# Avaliação no conjunto de validação
print("\nAvaliação no conjunto de validação completo:")
X_val_full = torch.cat([batch[0] for batch in val_loader], dim=0)
y_val_full = torch.cat([batch[1] for batch in val_loader], dim=0)

with torch.no_grad():
    y_pred_val = model(X_val_full)
    y_pred_val_labels = (y_pred_val >= 0.5).float()
    acuracia_val = (y_pred_val_labels == y_val_full).float().mean()
    print(f"Acurácia no conjunto de validação: {acuracia_val:.4f}")

    tp_val = ((y_pred_val_labels == 1) & (y_val_full == 1)).sum()
    tn_val = ((y_pred_val_labels == 0) & (y_val_full == 0)).sum()
    fp_val = ((y_pred_val_labels == 1) & (y_val_full == 0)).sum()
    fn_val = ((y_pred_val_labels == 0) & (y_val_full == 1)).sum()

    print("\nMatriz de Confusão (Validação):")
    print(f"                Previsto 0    Previsto 1")
    print(f"Real 0      {tn_val.item():>10}    {fp_val.item():>10}")
    print(f"Real 1      {fn_val.item():>10}    {tp_val.item():>10}")

# Avaliação do conjunto de teste
print("\nAvaliação no conjunto de teste:")
with torch.no_grad():
    y_pred_test = model(X_test)
    y_pred_labels = (y_pred_test >= 0.5).float()
    acuracia = (y_pred_labels == y_test).float().mean()
    print(f"Acurácia no conjunto de teste: {acuracia:.4f}")

    tp_test = ((y_pred_labels == 1) & (y_test == 1)).sum()
    tn_test = ((y_pred_labels == 0) & (y_test == 0)).sum()
    fp_test = ((y_pred_labels == 1) & (y_test == 0)).sum()
    fn_test = ((y_pred_labels == 0) & (y_test == 1)).sum()

    print("\nMatriz de Confusão (Teste):")
    print(f"                Previsto 0    Previsto 1")
    print(f"Real 0      {tn_test.item():>10}    {fp_test.item():>10}")
    print(f"Real 1      {fn_test.item():>10}    {tp_test.item():>10}")

# Cálculo do MSE para cada conjunto e sua impressão
mse_loss = nn.MSELoss()

with torch.no_grad():
    # MSE - Treinamento
    mse_treino = mse_loss(y_pred_train, y)
    print(f"\nMSE no conjunto de treinamento: {mse_treino.item():.6f}")

    # MSE - Validação
    mse_validacao = mse_loss(y_pred_val, y_val_full)
    print(f"MSE no conjunto de validação: {mse_validacao.item():.6f}")

    # MSE - Teste
    mse_teste = mse_loss(y_pred_test, y_test)
    print(f"MSE no conjunto de teste: {mse_teste.item():.6f}")

# Função para calcular precisão, recall e F1-score
def calcular_metricas(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision.item(), recall.item(), f1.item()

# Métricas adicionais no conjunto de TREINAMENTO
precision_train, recall_train, f1_train = calcular_metricas(tp_train, fp_train, fn_train)
print("\nMétricas (Treinamento):")
print(f"Precisão: {precision_train:.4f}")
print(f"Revocação (Recall): {recall_train:.4f}")
print(f"F1-Score: {f1_train:.4f}")

# Métricas adicionais no conjunto de VALIDAÇÃO
precision_val, recall_val, f1_val = calcular_metricas(tp_val, fp_val, fn_val)
print("\nMétricas (Validação):")
print(f"Precisão: {precision_val:.4f}")
print(f"Revocação (Recall): {recall_val:.4f}")
print(f"F1-Score: {f1_val:.4f}")

# Métricas adicionais no conjunto de TESTE
precision_test, recall_test, f1_test = calcular_metricas(tp_test, fp_test, fn_test)
print("\nMétricas (Teste):")
print(f"Precisão: {precision_test:.4f}")
print(f"Revocação (Recall): {recall_test:.4f}")
print(f"F1-Score: {f1_test:.4f}")

