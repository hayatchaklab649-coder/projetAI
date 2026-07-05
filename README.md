# Mini projet IA & NLP - Analyse des sentiments

Application web locale pour explorer un jeu de donnees d'avis clients et predire le sentiment d'un nouveau texte.

## Fonctionnalites

- Exploration de donnees : volume, repartition des sentiments, longueurs des textes, mots frequents.
- Pipeline NLP : nettoyage, stop words francais, vectorisation TF-IDF.
- Modele IA : regression logistique pour classer `positif`, `neutre`, `negatif`.
- Interface web : tableau de bord, prediction interactive, apercu du dataset, metriques du modele.

## Structure

```text
.
|-- app.py
|-- data/
|   `-- avis_clients.csv
|-- static/
|   |-- app.js
|   `-- styles.css
|-- templates/
|   `-- index.html
|-- requirements.txt
`-- README.md
```

## Installation

Si Python est installe sur ta machine :

```bash
pip install -r requirements.txt
python app.py
```

Puis ouvre :

```text
http://127.0.0.1:8000
```

## Deploiement

Le projet est pret pour un deploiement sur Render, Railway ou toute plateforme Python qui lance une commande web.

### Option Render

1. Mets le projet sur GitHub.
2. Va sur Render et cree un nouveau `Web Service`.
3. Connecte ton depot GitHub.
4. Utilise ces parametres :
   - Build command : `pip install -r requirements.txt`
   - Start command : `python app.py`
5. Render fournit automatiquement la variable `PORT`, et l'application l'utilise.

Le fichier `render.yaml` permet aussi de declarer le service automatiquement depuis Render.

### Option Railway

1. Cree un nouveau projet Railway.
2. Importe le depot GitHub.
3. Railway detecte Python.
4. Commande de demarrage : `python app.py`

En local, l'application utilise `127.0.0.1:8000`. En production, utilise :

```bash
HOST=0.0.0.0 PORT=8000 python app.py
```

## Jeu de donnees

Le fichier `data/avis_clients.csv` contient des avis clients en francais avec trois classes :

- `positif`
- `neutre`
- `negatif`

Tu peux remplacer ce fichier par ton propre dataset si tu gardes au minimum ces colonnes :

- `text`
- `sentiment`

## Methodologie

1. Chargement des donnees.
2. Nettoyage du texte : minuscules, suppression des caracteres inutiles, retrait des stop words.
3. Vectorisation avec `TfidfVectorizer`.
4. Entrainement d'une `LogisticRegression`.
5. Evaluation avec accuracy, precision, recall, F1-score et matrice de confusion.
6. Prediction d'un sentiment sur un texte saisi dans l'interface.
