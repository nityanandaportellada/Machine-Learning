import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import matplotlib.patches as mpatches
import urllib.request
from sklearn.metrics import mean_squared_error

# Funções auxiliares
def relu(Z):
    return np.maximum(0, Z)

def relu_derivada(Z):
    return (Z > 0).astype(float)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Baixar dados
url1 = "https://www.dropbox.com/scl/fi/xtb93nozp3w58unrw49rz/dados_pessoas_treinamento.csv?rlkey=egd6cho4xuyslgrxdabcvsgqf&dl=1"
url2 = "https://www.dropbox.com/scl/fi/ecoop00zvu4r2bkao0f4e/dados_pessoas_teste.csv?rlkey=5blyg2y5ufjsq6vluplxeofvi&dl=1"
urllib.request.urlretrieve(url1, "dados_pessoas_treinamento.csv")
urllib.request.urlretrieve(url2, "dados_pessoas_teste.csv")

# Carregar dados
dfdados1 = pd.read_csv('dados_pessoas_treinamento.csv', header=None)
dfdados2 = pd.read_csv('dados_pessoas_teste.csv', header=None)

X_train = dfdados1.iloc[:, :-1].values
Y_train = dfdados1.iloc[:, -1].values.reshape(-1, 1)
X_test = dfdados2.iloc[:, :-1].values
Y_test = dfdados2.iloc[:, -1].values.reshape(-1, 1)

# Inicialização dos pesos
def inicializar_pesos(input_size, hidden_size1, hidden_size2):
    W1 = np.random.randn(input_size, hidden_size1) * 0.01
    b1 = np.zeros((1, hidden_size1))
    W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
    b2 = np.zeros((1, hidden_size2))
    W3 = np.random.randn(hidden_size2, 1) * 0.01
    b3 = np.zeros((1, 1))
    return W1, b1, W2, b2, W3, b3

# Forward pass
def forward_pass(X, W1, b1, W2, b2, W3, b3):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = relu(Z2)
    Z3 = A2.dot(W3) + b3
    A3 = sigmoid(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Custo
def compute_cost(Y, A3):
    A3 = np.clip(A3, 1e-10, 1 - 1e-10)
    return -np.mean(Y * np.log(A3) + (1 - Y) * np.log(1 - A3))

# Backpropagation
def backward_pass(X, Y, Z1, A1, Z2, A2, Z3, A3, W2, W3):
    m = X.shape[0]
    dZ3 = A3 - Y
    dW3 = (A2.T @ dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_derivada(Z2)
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivada(Z1)
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3

# Treinamento
def treinar_RNA(X, Y, hidden_size1=6, hidden_size2=4, num_epocas=1000, alpha=0.01):
    input_size = X.shape[1]
    W1, b1, W2, b2, W3, b3 = inicializar_pesos(input_size, hidden_size1, hidden_size2)
    historico_custo = []

    for _ in range(num_epocas):
        Z1, A1, Z2, A2, Z3, A3 = forward_pass(X, W1, b1, W2, b2, W3, b3)
        custo = compute_cost(Y, A3)
        historico_custo.append(custo)
        dW1, db1, dW2, db2, dW3, db3 = backward_pass(X, Y, Z1, A1, Z2, A2, Z3, A3, W2, W3)

        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        W3 -= alpha * dW3
        b3 -= alpha * db3

    return W1, b1, W2, b2, W3, b3, historico_custo

# Predição
def predizer(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_pass(X, W1, b1, W2, b2, W3, b3)
    return (A3 >= 0.5).astype(int)

# Contribuição dos parâmetros
def contribuicao_parametros_csv(x, Y_real, W1, b1, W2, b2, W3, b3, nome_arquivo='contribuicoes.csv'):
    Z1, A1, Z2, A2, Z3, A3 = forward_pass(x.reshape(1, -1), W1, b1, W2, b2, W3, b3)
    y_pred = int(A3 >= 0.5)
    registros = []

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            registros.append({
                'Parâmetro': f'θ_{i+1}_{j+1}',
                'Peso': float(np.round(W1[i, j], 4)),
                'Entrada x': float(np.round(x[i], 4)),
                'Contribuição': float(np.round(W1[i, j] * x[i], 4)),
                'Classe prevista': y_pred,
                'Classe real': int(Y_real),
                'Probabilidade': float(np.round(A3[0, 0], 4))
            })

    df_contrib = pd.DataFrame(registros)
    df_contrib.to_csv(nome_arquivo, index=False)
    print(f"Arquivo '{nome_arquivo}' salvo com sucesso com {len(df_contrib)} contribuições.")

# Multi-tabela HTML
def multi_table(table_list):
    html_code = (
        '<table style="margin: 0 auto;"><tr style="vertical-align:top;background-color:white;">' +
        ''.join(['<td style="vertical-align:top;">' + table._repr_html_() + '</td>' for table in table_list]) +
        '</tr></table>'
    )
    return HTML(html_code)

# Impressão de parâmetros
def imprimir_parametros(W1, b1, W2, b2, W3, b3, numEpocas, alpha, CustoFinal, acuracia_treinamento):
    styles = [
        dict(selector="tr:hover", props=[("background-color", "#ffff99")]),
        dict(selector="td", props=[("font-size", "120%"), ("text-align", "center")]),
        dict(selector="th", props=[("font-size", "120%"), ("text-align", "left")]),
        dict(selector="caption", props=[("caption-side", "bottom")])
    ]

    def tabela_pesos(W, nome):
        ids = range(W.size)
        return pd.DataFrame({
            r'Vetor $\theta$': np.char.mod(r'$\theta_{%d}$', ids),
            'Valor do parâmetro': W.flatten()
        }).style.set_table_styles(styles).set_caption(nome)

    def tabela_bias(b, nome):
        return pd.DataFrame({
            r'Vetor $\theta$': [r'$\theta_{b}$'] * b.shape[1],
            'Valor do parâmetro': b.flatten()
        }).style.set_table_styles(styles).set_caption(nome)

    table_params = pd.DataFrame({
        'Valores': [numEpocas, alpha, CustoFinal, f"{acuracia_treinamento:.2f}%"]
    }, index=['Número de épocas', 'Taxa de aprendizagem', 'Custo final:', 'Acurácia']).style.set_table_styles(styles).set_caption("Parâmetros de Treinamento")

    display(multi_table([
        tabela_pesos(W1, "Pesos da camada 1 (Input -> Hidden 1)"),
        tabela_bias(b1, "Bias da camada 1"),
        tabela_pesos(W2, "Pesos da camada 2 (Hidden 1 -> Hidden 2)"),
        tabela_bias(b2, "Bias da camada 2"),
        tabela_pesos(W3, "Pesos da camada 3 (Hidden 2 -> Output)"),
        tabela_bias(b3, "Bias da camada 3"),
        table_params
    ]))

# === Exemplo de treinamento ===
W1, b1, W2, b2, W3, b3, historico_custo = treinar_RNA(X_train, Y_train, hidden_size1=6, hidden_size2=4, num_epocas=1000, alpha=0.01)
pred_train = predizer(X_train, W1, b1, W2, b2, W3, b3)
acuracia_treinamento = 100 * np.mean(pred_train == Y_train)
CustoFinal = historico_custo[-1]
imprimir_parametros(W1, b1, W2, b2, W3, b3, 1000, 0.01, CustoFinal, acuracia_treinamento)

def plotar_custo(historico_custo):
    plt.figure(figsize=(8, 5))
    plt.plot(historico_custo, label='Custo', color='blue')
    plt.title('Evolução do Custo ao Longo das Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Custo')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Após treinamento
plotar_custo(historico_custo)

# Predição contínua para MSE
_, _, _, _, _, A3_train = forward_pass(X_train, W1, b1, W2, b2, W3, b3)
mse_train = mean_squared_error(Y_train, A3_train)
print(f"Erro Quadrático Médio (MSE) no treinamento: {mse_train:.6f}")

