# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# Charger les données à partir du fichier CSV
data = pd.read_csv('ddo.csv')

# Convertir la colonne 'date' en une représentation numérique
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour
data['minute'] = data['date'].dt.minute

# Supprimer la colonne 'date' d'origine
data = data.drop('date', axis=1)

# Encodage one-hot des variables catégorielles
data = pd.get_dummies(data, columns=['WeekStatus', 'Day_of_week', 'Load_Type'], drop_first=True)

# Séparation des caractéristiques (X) et de la variable cible (y)
X = data.drop('Usage_kWh', axis=1)  # Remplacez 'Variable_Cible' par le nom de votre variable cible
y = data['Usage_kWh']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création et entraînement du modèle d'arbre de décision pour la régression
regressor = DecisionTreeRegressor()  # Créez une instance du modèle de régression d'arbre de décision
regressor.fit(X_train, y_train)  # Entraînez le modèle sur l'ensemble d'entraînement

# Prédictions sur l'ensemble de test
y_pred = regressor.predict(X_test)

# Ajouter la colonne des prédictions au dataframe des données de test
X_test['Predicted_Usage_kWh'] = y_pred

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
correlation, _ = pearsonr(y_test, y_pred)

print(f'Erreur quadratique moyenne (MSE) : {mse:.2f}')
print(f'Coefficient de détermination (R^2) : {r2:.2f}')
print(f'Score de prédiction : {correlation:.2f}')

# Afficher les données de test avec les prédictions
print(X_test)
