import keras_tuner as kt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, BatchNormalization
import kagglehub

# Set seed for reproducibility
np.random.seed(5)
tf.random.set_seed(5)

# Load dataset
path = kagglehub.dataset_download("kumarajarshi/life-expectancy-who")
data = pd.read_csv(path+'/Life Expectancy Data.csv')

data = data.dropna()

# Preprocess data
data['Country'] = data['Country'].astype('category').cat.codes
data['Status'] = data['Status'].astype('category').cat.codes  # Convert 'Status' column to numeric codes
X = data.drop(columns=['Life expectancy '])
y = data['Life expectancy ']

# Ensure 'Country' is treated as a separate feature for embedding
X['Country'] = X['Country'].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Ensure 'Country' is not scaled
X_train_scaled[:, X.columns.get_loc('Country')] = X_train['Country'].values
X_test_scaled[:, X.columns.get_loc('Country')] = X_test['Country'].values

# Scale target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Define model building function for Keras Tuner
def build_model(hp):
    input_country = Input(shape=(1,), name='Country')
    input_features = Input(shape=(X_train_scaled.shape[1],), name='features')

    embedding = Embedding(input_dim=len(data['Country'].unique()), 
                          output_dim=hp.Int('embedding_output_dim', min_value=5, max_value=50, step=5))(input_country)
    flatten = Flatten()(embedding)

    concatenated = Concatenate()([flatten, input_features])
    x = concatenated

    for i in range(hp.Int('num_layers', 1, 3)):
        x = Dense(units=hp.Int(f'units_{i}', min_value=16, max_value=128, step=16),
                  activation=hp.Choice('activation', values=['relu', 'tanh']))(x)
        x = BatchNormalization()(x)

    output = Dense(1, activation='linear', name='output_layer')(x)

    model = Model(inputs=[input_country, input_features], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='mse',
                  metrics=['mae'])
    return model

# Initialize Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='life_expectancy'
)

# Search for the best hyperparameters
tuner.search([X_train['Country'], X_train_scaled], y_train_scaled, epochs=50, validation_split=0.2, verbose=1)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters and train it
model = build_model(best_hps)
history = model.fit([X_train['Country'], X_train_scaled], y_train_scaled, epochs=50, validation_split=0.2, verbose=1)

# Predict and evaluate
y_pred_scaled = model.predict([X_test['Country'], X_test_scaled])
y_pred = scaler_y.inverse_transform(y_pred_scaled)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
