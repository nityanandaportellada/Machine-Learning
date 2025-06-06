#importando conjuntos necessários
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt

# Download de ambas as bases de dados
url_treinamento = "https://www.dropbox.com/scl/fi/o0rynsfm9qm925niir5un/dados_caracteres_treinamento_RN.csv?rlkey=3vn4nx5in0ov3fdgqojhrl3c5&dl=1"
url_teste = "https://www.dropbox.com/scl/fi/g52nf9ybn2h6w3xsp8opk/dados_caracteres_teste_RN.csv?rlkey=4u55w4as35ugsbtugz4whe3s0&dl=1"
urllib.request.urlretrieve(url_treinamento, "dados_caracteres_treinamento_RN.csv")
urllib.request.urlretrieve(url_teste, "dados_caracteres_teste_RN.csv")

# Carregando e convertendo os dados
df_treino = pd.read_csv("dados_caracteres_treinamento_RN.csv", header=None)
df_teste = pd.read_csv("dados_caracteres_teste_RN.csv", header=None)

X = torch.tensor(df_treino.iloc[:, :-1].values, dtype=torch.float32)
y = torch.tensor(df_treino.iloc[:, -1].values - 1, dtype=torch.long)
X_test = torch.tensor(df_teste.iloc[:, :-1].values, dtype=torch.float32)
y_test = torch.tensor(df_teste.iloc[:, -1].values - 1, dtype=torch.long)

# Função para dividir o banco de dado de treinamento em dado de treino e dado de validação,
# utilizou-se a proporção 0.4 e 0.6 diante do melhor resultado para este modelo e para diminuir
# o overfitting
def dividir_treino_validacao(X, y, proporcao_treino=0.4):
    dataset = TensorDataset(X, y)
    tamanho_treino = int(proporcao_treino * len(dataset))
    tamanho_validacao = len(dataset) - tamanho_treino
    return random_split(dataset, [tamanho_treino, tamanho_validacao])

# Implementação do modelo de rede neural. Após testes optou-se por uma camada oculta com 100 neuronios
class ClassificadorMulticlasse(nn.Module):
    def __init__(self, input_size, num_classes, dropout_prob=0.5):
        super(ClassificadorMulticlasse, self).__init__()
        self.hidden = nn.Linear(input_size, 100)
        self.dropout = nn.Dropout(dropout_prob)  # Camada de dropout aplicada para diminuir overfitting
        self.output = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.dropout(x)  # Aplica o dropout após a ativação ReLU
        x = self.output(x) #utilizado self uma vez que a softmax esta inclusa no backpropagation
        return x

# Preparando os dados para uso, optou-se por batch size de 16 diante do melhor resultado para o modelo
dataset_treino, dataset_validacao = dividir_treino_validacao(X, y)
train_loader = DataLoader(dataset_treino, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset_validacao, batch_size=16)

input_size = X.shape[1]
num_classes = len(torch.unique(y))  # gerando o Número de classes
model = ClassificadorMulticlasse(input_size, num_classes)

# Criação dos Hiperparâmetros
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4) #regularização usando L2
epochs = 500

# Criação de vetores para armazenar os valores de perda entre treino e validação
losses_treino = []
losses_validacao = []

# Realização do Treinamento com validação por época
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

# Gráfico de loss em relação à época
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
    _, y_pred_train_labels = torch.max(y_pred_train, 1)
    acuracia_treino = (y_pred_train_labels == y).float().mean()
    print(f"Acurácia no conjunto de treinamento: {acuracia_treino:.4f}")

    # Cálculo da matriz de confusão
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y.view(-1), y_pred_train_labels.view(-1)):
        conf_matrix[t.long(), p.long()] += 1

    print("\nMatriz de Confusão (Treinamento):")
    print(conf_matrix)

    # Cálculo de métricas por classe
    for classe in range(num_classes):
        tp = conf_matrix[classe, classe].item()
        fp = conf_matrix[:, classe].sum().item() - tp
        fn = conf_matrix[classe, :].sum().item() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        print(f"\nClasse {classe + 1}:")
        print(f"  Precisão: {precision:.4f}")
        print(f"  Revocação (Recall): {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")

# Avaliação no conjunto de validação
print("\nAvaliação no conjunto de validação completo:")
X_val_full = torch.cat([batch[0] for batch in val_loader], dim=0)
y_val_full = torch.cat([batch[1] for batch in val_loader], dim=0)

with torch.no_grad():
    y_pred_val = model(X_val_full)
    _, y_pred_val_labels = torch.max(y_pred_val, dim=1)
    acuracia_val = (y_pred_val_labels == y_val_full).float().mean()
    print(f"Acurácia no conjunto de validação: {acuracia_val:.4f}")

    # Matriz de confusão
    conf_matrix_val = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_val_full.view(-1), y_pred_val_labels.view(-1)):
        conf_matrix_val[t.long(), p.long()] += 1

    print("\nMatriz de Confusão (Validação):")
    print(conf_matrix_val)

    # Cálculo de métricas por classe
    for classe in range(num_classes):
        tp = conf_matrix_val[classe, classe].item()
        fp = conf_matrix_val[:, classe].sum().item() - tp
        fn = conf_matrix_val[classe, :].sum().item() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        print(f"\nClasse {classe + 1}:")
        print(f"  Precisão: {precision:.4f}")
        print(f"  Revocação (Recall): {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")

# Avaliação no conjunto de teste
print("\nAvaliação no conjunto de teste:")
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    _, y_pred_test_labels = torch.max(y_pred_test, dim=1)
    acuracia_teste = (y_pred_test_labels == y_test).float().mean()
    print(f"Acurácia no conjunto de teste: {acuracia_teste:.4f}")

    # Matriz de confusão
    conf_matrix_teste = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_test.view(-1), y_pred_test_labels.view(-1)):
        conf_matrix_teste[t.long(), p.long()] += 1

    print("\nMatriz de Confusão (Teste):")
    print(conf_matrix_teste)

    # Cálculo de métricas por classe
    for classe in range(num_classes):
        tp = conf_matrix_teste[classe, classe].item()
        fp = conf_matrix_teste[:, classe].sum().item() - tp
        fn = conf_matrix_teste[classe, :].sum().item() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        print(f"\nClasse {classe + 1}:")
        print(f"  Precisão: {precision:.4f}")
        print(f"  Revocação (Recall): {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")

