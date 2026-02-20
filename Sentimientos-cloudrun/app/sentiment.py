from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

textos = [
    "me encanta este producto", "excelente servicio", "muy buena experiencia",
    "todo perfecto", "gran trabajo", "me gustó mucho", "fantástico resultado",
    "muy recomendable", "estoy satisfecho", "buenísimo",

    "es normal", "está bien supongo", "nada especial", "aceptable", "más o menos",
    "no está mal", "regular servicio", "puede mejorar", "está decente", "no es la gran cosa",

    "esto es horrible", "eres un idiota", "pésimo servicio", "no me gustó para nada",
    "terrible experiencia", "asco total", "qué basura", "muy malo", "es un desastre", "odio esto"
]

clases = ["normal"] * 10 + ["regular"] * 10 + ["toxico"] * 10

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos).toarray()

encoder = LabelEncoder()
y = encoder.fit_transform(clases)

model = Sequential()
model.add(Dense(16, input_dim=X.shape[1], activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=100, verbose=0)

def predict(text):
    vec = vectorizer.transform([text]).toarray()
    pred = model.predict(vec, verbose=0)
    clase = encoder.inverse_transform([np.argmax(pred)])
    return clase[0]