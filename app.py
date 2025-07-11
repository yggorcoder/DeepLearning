import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import models
from tensorflow.keras import layers

# alterar a geração de números aleatórios, para tornar o código reproduzível
np.random.seed(0)

#Definir o número de features
number_of_features = 1000

# Carregar o conjunto de dados
(data_train, target_train), (data_test, target_test) = keras.datasets.imdb.load_data(
    num_words=number_of_features)

#Fazer com que cada observação tenha no máximo 400 features
features_train = sequence.pad_sequences(data_train, maxlen=400)
features_test = sequence.pad_sequences(data_test, maxlen=400)

#Iniciar a rede neural
network = models.Sequential()

# Adicionar uma camada do tipo embedding
network.add(layers.Embedding(input_dim=number_of_features, output_dim=128))

# Adicionar uma camada long short-term memory layer com 128 unidades
network.add(layers.LSTM(units=128))

# Adicionar uma camada totalmente conectada com a função de ativação sigmoid
network.add(layers.Dense(units=1, activation="sigmoid"))

# Compilar a rede neural
network.compile(loss="binary_crossentropy", # entropia cruzada
                optimizer="Adam", # otimizador do tipo Adam
                metrics=["accuracy"]) # acurácio como métrica de desempenho

# Treinar a rede neural
history = network.fit(features_train, # Features
                      target_train, # Target
                      epochs=3, # Número de épocas
                      batch_size=1000, # Número de observações por lote
                      validation_data=(features_test, target_test)) # dados de teste