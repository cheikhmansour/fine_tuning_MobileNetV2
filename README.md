Pré-requis
Avant d'exécuter le projet, assurez-vous d'avoir les outils suivants installés :

Python 3.x
TensorFlow
Keras
Matplotlib
NumPy
Pandas
scikit-learn
autres librairies définies dans requirements.txt

Préparer les données : Placez le dataset d'images dans le dossier dataset/ et assurez-vous que les images sont correctement organisées par classe.

Entraînement du modèle : Lancez le script d'entraînement avec la commande suivante : executer les cellules dans le fichier exam_fruit360.ipynb

Résultats
Accuracy d'entraînement : 93,94% après 8 epochs
Accuracy de validation : 52,50% après 8 epochs
Perte d'entraînement : 0,20
F1-Score : Variabilité selon les classes avec certaines catégories obtenant des scores plus élevés que d'autres.
Quelques classes comme "Banana" ou "Tomato" ont des performances particulièrement bonnes, tandis que d'autres, comme "Pomegranate" ou "Mango Red", montrent de moins bons résultats.


Améliorations futures
Utiliser l'augmentation des données pour améliorer les performances sur les classes sous-représentées.
Ajuster les hyperparamètres pour optimiser les performances du modèle.
Tester des modèles différents et les comparer pour voir si des architectures plus complexes peuvent donner de meilleurs résultats.
