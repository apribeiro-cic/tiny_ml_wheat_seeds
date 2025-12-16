# Classificador de Trigo com TinyML 🌾

Este projeto foi desenvolvido para a placa **BitDogLab**, baseada no microcontrolador **Raspberry Pi Pico W**, com o objetivo de criar um sistema embarcado inteligente, capaz de realizar a classificação de um dataset tabular simples utilizando a biblioteca **TensorFlow Lite Micro (TFLM)**. O sistema executa a inferência em um dataset completo embarcado e gera uma **Matriz de Confusão** via Serial para comparação com os resultados obtidos no computador.


---

## 📌 Sobre o Projeto

O Classificador TinyML foi desenvolvido como projeto prático da 2ª fase da residência tecnológica **EmbarcaTech**.

O objetivo principal foi cumprir o pipeline completo de desenvolvimento:
1.  **Treinamento:** Definição e treino de uma Rede Neural (MLP) no Google Colab.
2.  **Conversão:** Otimização do modelo para `.tflite` e conversão para array C (`.h`).
3.  **Deploy:** Implementação do firmware em C++ para o RP2040.
4.  **Validação:** Comparação dos resultados obtidos no PC vs. Placa (Matriz de Confusão).

---

## 🧠 Como funciona

O firmware atua como um "auditor" do modelo de Inteligência Artificial. Ele não aguarda interação do usuário; ao ser iniciado, ele processa todas as amostras disponíveis e apresenta o relatório final.

### 🧪 Dataset Embarcado

- **Fonte de Dados:** O arquivo `wheat_seeds_dataset.h` contém **210 amostras** estáticas (hardcoded) do dataset original.
- **Estrutura:** Cada amostra possui 7 características geométricas (Área, Perímetro, Compacidade, etc.) e seu rótulo verdadeiro (Classe 0, 1 ou 2).
- **Normalização:** O microcontrolador aplica a mesma normalização (StandardScaler) utilizada no treinamento (`mean` e `std_dev`) antes de processar cada entrada.

### 🤖 Motor de Inferência

- **TensorFlow Lite Micro:** A biblioteca carrega o modelo otimizado e executa a classificação para cada uma das 210 amostras sequencialmente.
- O sistema compara a **Classe Predita** (saída do modelo) com a **Classe Real** (gabarito).

### 📊 Saída de Dados (Serial)

Ao final do processamento, o sistema imprime via USB (Serial Monitor):
1. **Log das primeiras 10 inferências** (para inspeção visual das probabilidades).
2. **Matriz de Confusão (3x3):** Tabela cruzando *Real* vs. *Predito*.
3. **Acurácia Final:** Porcentagem de acertos sobre o total de amostras.

---

## 📁 Utilização

Atendendo aos requisitos de organização da 2ª fase da residência, o arquivo `CMakeLists.txt` está configurado para facilitar a importação do projeto no Visual Studio Code. 

Siga as instruções:

1. Na barra lateral, clique em **Raspberry Pi Pico Project** e depois em **Import Project**.

   ![image](https://github.com/user-attachments/assets/4b1ed8c7-6730-4bfe-ae1f-8a26017d1140)

2. Selecione o diretório do projeto e clique em **Import** (utilizando a versão **2.1.1** ou superior do Pico SDK).

   ![image](https://github.com/user-attachments/assets/be706372-b918-4ade-847e-12706af0cc99)

3. Agora, basta **compilar** e **rodar** o projeto, com a placa **BitDogLab** conectada. Abra o Monitor Serial para acompanhar os logs de inferência.

---

## 📜 Créditos e Bibliotecas

Este projeto utiliza a biblioteca [pico-tflmicro](https://github.com/raspberrypi/pico-tflmicro), que é um port oficial do TensorFlow Lite Micro para o RP2040, licenciado sob **Apache 2.0**.

Os dados utilizados para treinamento são provenientes do [UCI Machine Learning Repository - Wheat Seeds Dataset](https://archive.ics.uci.edu/dataset/236/seeds).
