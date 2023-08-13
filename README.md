# Previsão de Degradação do Pneu com Random Forest

Este é um pequeno script em Python que utiliza o algoritmo Random Forest para prever quantas voltas um pneu pode durar com base em dados de pressão do pneu e temperatura da pista.

## Requisitos

- Python 3.x
- Bibliotecas: `numpy`, `scikit-learn`

Certifique-se de que você possui as bibliotecas necessárias instaladas. Você pode instalá-las usando o seguinte comando:

```bash
pip install numpy scikit-learn
```

## Uso
1 - Clone este repositório ou copie o código para o seu ambiente Python.

2 - Execute o script predict_tire_degradation.py:

```bash
python predict_tire_degradation.py
```

Insira a pressão do pneu e a temperatura da pista quando solicitado.

O script treinará um modelo de regressão Random Forest com os dados de exemplo e fará previsões para as entradas do usuário.
