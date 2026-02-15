# Base officielle TensorFlow GPU
FROM tensorflow/tensorflow:2.16.1-gpu

# Crée un répertoire de travail dans le conteneur
WORKDIR /workspace

# Copie les dépendances
COPY requirements.txt .

# Installe les dépendances via pip
RUN pip install --no-cache-dir -r requirements.txt

# Installation des modèles spaCy
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl \
    fr-core-news-sm \
    es-core-news-sm \
    https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-3.7.0/nl_core_news_sm-3.7.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/it_core_news_sm-3.7.0/it_core_news_sm-3.7.0-py3-none-any.whl

# Expose le port Jupyter
EXPOSE 8888

# Dossier de travail par défaut
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

