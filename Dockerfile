# Base officielle TensorFlow GPU
FROM tensorflow/tensorflow:2.16.1-gpu

# Crée un répertoire de travail dans le conteneur
WORKDIR /workspace

# Copie les dépendances
COPY requirements.txt .

# Installe les dépendances via pip (inclut les modèles spaCy)
RUN pip install --no-cache-dir -r requirements.txt

# Expose le port Jupyter
EXPOSE 8888

# Dossier de travail par défaut
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

