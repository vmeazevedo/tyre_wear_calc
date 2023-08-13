import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(X_train, y_train, num_estimators=100):
    """Treina um modelo de regressão com Random Forest."""
    model = RandomForestRegressor(n_estimators=num_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_laps(model, user_input):
    """Faz a previsão do número de voltas com base nos dados do usuário."""
    predicted_laps = model.predict([user_input])
    return predicted_laps[0]

def main():
    # Dados de exemplo (substitua isso pelos seus dados reais)
    tire_pressure = np.array([30.5, 29.8, 31.2, 30.0])
    track_temperature = np.array([25.0, 28.0, 30.0, 27.5])
    laps_before_change = np.array([20, 25, 18, 22])

    # Combina as características em uma matriz
    X = np.column_stack((tire_pressure, track_temperature))

    # Variável alvo
    y = laps_before_change

    # Divide os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treina o modelo
    model = train_model(X_train, y_train)

    # Faz previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Calcula o RMSE (Erro Quadrático Médio)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Erro Quadrático Médio: {rmse:.3f}")

    # Exemplo de previsão para entrada do usuário
    user_input = [30.3, 28.5]
    predicted_laps = predict_laps(model, user_input)
    print(f"Laps Previstas: {predicted_laps:.2f}")

if __name__ == "__main__":
    main()
