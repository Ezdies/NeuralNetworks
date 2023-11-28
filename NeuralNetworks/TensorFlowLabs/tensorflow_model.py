from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

keras = tf.keras

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Definicja modelu
model = keras.models.Sequential([
 tf.keras.layers.Dense(16, activation='relu', 
input_shape=(X_train.shape[1],)),
 tf.keras.layers.Dense(1, activation='sigmoid') # Sigmoid dla klasyfikacji binarnej
])

#Kompilowanie modelu

model.compile(optimizer='adam',
 loss='binary_crossentropy',
 metrics=['accuracy'])

#Trenowanie modelu

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, 
y_test))

#Ewaluacja modelu

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Strata Testowa: {loss}, Dokładność Testowa: {accuracy}")

#Analiza wyników


# Wykres dokładności
plt.plot(history.history['accuracy'], label='dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='dokładność walidacyjna')
plt.title('Dokładność w czasie')
plt.ylabel('Dokładność')
plt.xlabel('Epoka')
plt.legend()
plt.show()




