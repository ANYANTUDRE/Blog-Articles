---
title: "Le Surapprentissage ou Surajustement (overfitting) en Machine Learning"
datePublished: Wed Jan 17 2024 05:58:14 GMT+0000 (Coordinated Universal Time)
cuid: clrhdgeu8001209jx5cjk0wp8
slug: le-surapprentissage-ou-surajustement-overfitting-en-machine-learning
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1705471135875/4dc9f028-82a7-446f-ba46-c10202e9d951.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1705471055241/70aa508a-f8be-4d8a-bc0b-06791943c79f.png
tags: optimization, data-science, machine-learning, deep-learning, overfitting

---

![Source: Wikipedia — Overfitting](https://cdn-images-1.medium.com/max/800/1*3wSW_r5fSSTzxCLurfk5Zg.png align="center")

Nous, les humains, avons très souvent tendance à surgénéraliser les faits dont nous sommes témoin. Par exemple, lorsque nous arrivons tout nouvellement en terre étrangère et qu’un taximan ou un boutiquier nous escroque, nous serions, à priori, tentés dans cette situation d’affirmer que tous les taximen ou les boutiquiers de ce pays sont des “voleurs”. Cette tendance à faire des inductions hatives, c’est-à-dire à déduire une règle générale à partir d’une simple observation conduit très souvent à des conclusions incorrectes.

Eh bien, figurez vous que tout comme nous, c’est malheureusement un piège dans lequel tombent également les algorithmes de Machine Learning quand nous ne sommes pas vigilants lors de leur entrainement. ***C’est cette situation de surgénéralisation qu’on appelle Overfitting (ou surapprentissage ou encore surajustement en français).***

Dans cet article, nous allons aborder en détails, les points suivants :

1. Définition de l’Overfitting
    
2. Les causes de l’Overfitting
    
3. Un cas pratique d’Overfitting
    
4. Une explication plus courante de l’Overfitting
    
5. Comment éviter l’Overfitting?
    
6. Le principe de simplicité ou Razoir d’Occam
    

### 1\. Définition de l’Overfitting

> *On parle d’Overfitting quand un algorithme de Machine Learning performe très très bien sur les données d’apprentissage qu’on lui fournit (training set) mais obtient relativement de mauvaises performances lorqu’il est testé sur de nouveaux exemples (test set) qu’il n’a jamais vus.*

En effet, lorsqu’un modèle est trop complexe et s’ajuste trop étroitement aux données d’entraînement, capturant même le bruit et les erreurs dans ces données, cela peut conduire à une ***mauvaise généralisation du modèle sur de nouvelles données, car il est trop spécialisé pour les données d’entraînement spécifiques.***

En francais facile, il y a un risque de tirer des conclusions trop hâtives à partir d’un ensemble limité d’informations, ce qui peut conduire à des erreurs lorsque ces conclusions sont appliquées à des situations plus générales ou à de nouvelles données.

![Source — https://www.freecodecamp.org/](https://cdn-images-1.medium.com/max/800/0*guBo5AYxPB4PI9o3.jpg align="left")

Nous verrons plutard une autre explication plus courante mais pour le moment la question que l’on doit se poser c’est ***qu’est-ce qui est à l’origine de l’overfitting?***

### 2\. Les causes de l’Overfitting (non exhaustif)

Généralement l’overfitting est lié soit au modèle ou soit aux données d’entraînement:

* **Un modèle un peu trop complexe** avec un grand nombre de paramètres ou une capacité d’ajustement élevée, peut être trop flexible et ajuster les données d’entraînement de manière trop précise.
    
* **Un manque (peu) de données d’entraînement** peut empêcher le modèle de capturer les variations réelles du dataset et donc il va se concentrer sur des détails spécifiques(souvent inutiles) qui ne se généralisent pas bien à de nouvelles données.
    
* **Une mauvaise qualité des données (présence de caractéristiques redondantes ou inutiles)** peut contribuer à l’overfitting car certaines informations peuvent être non pertinentes et ne généralisent pas vraiment à de nouvelles données.
    

C’est le cas par exemple lorqu’on a des valeurs abérrantes (outliers or noisy data points) dans le dataset.

![Source: TowardsAI](https://cdn-images-1.medium.com/max/800/1*AfKsExgtYR7yfPjZvciKXQ.png align="left")

Bien. Maintenant que nous avons une idée plus ou moins claire de ce que c’est que l’overfitting et de ses causes, voyons concrètement par un exemple comment il se manifeste.

### 3\. Un cas pratique d’Overfitting

Pour cet exemple, nous allons effectuer une classification binaire avec un ensemble de données sur la *prévision du taux d’attrition(désabonnement) de la clientèle des banques (Bank Churn Dataset)* que vous pouvez télécharcher sur Kaggle ou directement à partir du lien que je mettrai en fin d’article.

Commençons par importer les librairies de base dont nous aurons besoin. Nous importerons le reste au fur et à mesure que l’on avance.

```python
# importer les librairies necessaires
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ajuster la taille du texte sur les graphiques
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

# pour afficher les graphiques dans le notebook lui meme
%matplotlib inline
```

Puis on importe le fichier csv à l’aide de Pandas et on affiche les dix (10) premières lignes pour avoir une idée globale du dataset. Notez que j’utilise un notebook Kaggle pour cet exemple.

```python
# importer le fichier csv
df = pd.read_csv("/kaggle/input/playground-series-s4e1/train.csv")

# afficher les 10 premieres lignes du dataset
df.head(10)
```

![](https://cdn-images-1.medium.com/max/800/1*XsJrEBEWtwRB3rZFDEKZ0w.jpeg align="left")

***Petit apercu du Dataset — Bank Churn Dataset***

Nous avons en face un problème de classification binaire ou la variable cible (target) est la colonne ‘Exited’. Comme métrique d’évaluation, nous allons utiliser l’[*area under the ROC curve*](https://fr.wikipedia.org/wiki/Courbe_ROC) ou score AUC .   
Dans un but de simplicité, nous n’allons utiliser que quelques variables de ce dataset et on ne fera pas d’analyse exploratoire de données ou de feature engineering. Libre à vous de jetter un coup d’oeil rapide au dataset pour vous imprégner de la distribution des variables.

Par contre une étape essentielle avant d’atteindre notre but est de tranformer les variables catégorielles qu’on veut utiliser (notamment les variables ‘Geography’ et ‘Gender’) en variables numériques:

```python
# transformation des variables catégorielles en variables numériques
mapping_geography = {"France": 0, "Spain": 1, "Germany": 2}
mapping_gender    = {"Male": 0, "Female": 1}

# mapping
df.loc[:, "Geography"] = df.Geography.map(mapping_geography)
df.loc[:, "Gender"]    = df.Gender.map(mapping_gender)
```

Une fois les variables catégorielles mappées correctement(je ne sais pas si le verbe ‘mapper’ existe en francais:) ) et transformées, nous allons diviser le dataset en **train set**(données d’entrainement) et en **test set**(données d’évaluation). Retenez bien ces deux noms (train set et test set) car ils seront beaucoup utilisés par la suite.

```python
from sklearn.model_selection import train_test_split

# nous allons utiliser ces 10 colonnes
features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
 "HasCrCard", "IsActiveMember", "EstimatedSalary", "Geography", "Gender"]

X = df[features]    # features
y = df.Exited       # variable cible

# Division des données en train et test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2024)
```

Enfin nous pouvons entrainer un premier modèle de Machine Learning sur ce dataset. Pour simplifier, choisissons un algorithme simple et classique: un arbre de decision (decision tree) avec une profondeur maximale de 3 et les autres paramètres laissés à leurs valeurs par défaut:

```python
from sklearn.tree import DecisionTreeClassifier

# initialiser l'arbre de decision avec le parametre max_depth a 3
clf = DecisionTreeClassifier(max_depth=3)

# entrainer le modele sur les colonnes choisies plus haut
clf.fit(X_train, y_train)
```

```python
from sklearn.metrics import roc_auc_score # métrique AUC

# faire des predictions sur le training set et sur le test set
train_predictions = clf.predict(X_train)
test_predictions  = clf.predict(X_test)

# caluculer le score des predictions
train_auc = roc_auc_score(y_train, train_predictions)
test_auc  = roc_auc_score(y_test,  test_predictions)

print(f"Le score AUC sur le train set    ----> {train_auc}")
print(f"Le score AUC sur le test set     ----> {test_auc}")
```

![](https://cdn-images-1.medium.com/max/800/1*egeZEnOWbyVKkEpr_5nzAg.png align="center")

Scores AUC avec max\_depth=3

Les scores AUC d’entrainement et de test sont respectivement 68.47% et 68.79%(train\_auc ~ test\_auc). Maintenant, essayons d’augmenter la profondeur maximale à 7 et répétons la meme procédure. On obtient comme scores 74.88% pour le l’entrainement et 74.78% pour le test .

D’ailleurs, pourquoi ne pas judicieusement calculer ces scores pour différentes valeurs de max\_depth et tracer un graphique pour suivre leur évolution?

```python
# initialiser deux listes pour stocker les scores de train et test
train_auc_scores, test_auc_scores = list(), list()

# iterations pour quelques valeurs de la profondeur maximale(max_depth)
for depth in range(1, 40):
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    
    # predictions sur le training set et sur le test set
    train_predictions = clf.predict(X_train)
    test_predictions  = clf.predict(X_test)
    
    # caluculer les scores des predictions
    train_auc = roc_auc_score(y_train, train_predictions)
    test_auc  = roc_auc_score(y_test,  test_predictions)
    
    # ajouter les scores aux listes
    train_auc_scores.append(train_auc)
    test_auc_scores.append(test_auc)
```

```python
# tracer deux graphiques en utilisant matplotlib et seaborn
plt.figure(figsize=(10, 5))
sns.set_style("darkgrid")
plt.plot(train_auc_scores, label="train AUC score")
plt.plot(test_auc_scores, label="test AUC score")
plt.legend(loc="upper left", prop={'size': 15})
plt.xticks(range(0, 30, 3))
plt.xlabel("max_depth", size=20)
plt.ylabel("AUC score", size=20)
plt.show()
```

Nous constatons que le meilleur score pour le test set est obtenu lorsque *max\_depth* a une valeur approximative de 7. Au fur et à mesure que nous augmentons la valeur de ce paramètre, **le score du test set reste le même ou se dégrade tandis que le score du train continue d’augmenter.** Cela signifie que notre modèle continue donc d’apprendre les données d’entrainement avec l’augmentation de max\_depth, mais la performance sur les données de test ne s’améliore pas (enfin ne s’améliore plus disons).

**C’est ce qu’on appelle l’overfitting.**

![](https://cdn-images-1.medium.com/max/800/1*27jCo850JypxGwLnzX2Ezw.jpeg align="left")

Le modèle s’adapte parfaitement au train set mais donne des résultats très médiocres sur le test set. Cela signifie que le modèle apprendra bien les données d’apprentissage mais ne pourra pas généraliser sur des échantillons non vus. Dans l’ensemble de données ci-dessus, il est possible de construire un modèle avec un max\_depth très élevé qui obtiendra des résultats exceptionnels sur le train set, mais ce type de modèle n’est pas utile car il ne sera pas généralisable sur des échantillons non vus.

### 4\. Une explication plus courante de l’Overfitting

Chaque fois que nous entrainons un réseau de neuronnes (neural network), nous devons surveiller l’évolution du coût (loss en anglais) durant l’entrainement à la fois pour le train set et le test set. Si nous avons un très grand réseau pour un ensemble de données qui est très petit (c’est-à-dire un très petit nombre d’échantillons), nous observerons que les coûts du train et du test set diminuent au fur et à mesure de l’apprentissage.

Cependant, à un moment donné, le coût du test set atteindra son minimum, et après cela, commencera à augmenter même si le coût du train set continue de diminuer. Nous devons ainsi arrêter l’entrainement lorsque le coût du test/validation set atteint sa valeur minimale.

![Source: MLearning.ai](https://cdn-images-1.medium.com/max/800/0*p5gD86LO2m3B7C8R.png align="left")

**C’est l’explication la plus courante que vous trouverez sur l’Overfitting.**

Bien, nous avons assez exposé le problème de l’overfitting en Machine Learning, à présent, ***place aux mesures pour éviter l’overfitting.***

### 5\. Comment éviter l’overfitting?

Quelques mesures que l’on peut mettre en place pour lutter contre l’overfitting :

* **Toujours chercher à acquérir plus de données:** plus de données, c’est synomyne de plus d’informations pour les modèles qui pourront généraliser correctement.
    
* **S’assurer de la bonne qualité des données** que l’on fournit aux modèles(Garbage In, Garbage Out). La qualité des données est probablement la clé pour avoir de bonnes performances et eviter l’overfitting. Quelques idées:
    

✔ éliminer les caractéristiques redondantes, inutiles ou fortement corrélées pour diminuer le bruit (noise)

✔ fixer les potentielles erreurs dans le dataset

✔ traiter correctement les valeurs abérrantes (outliers) etc…

* **Simplifier au maximum le modèle** (on en reparlera dans la section suivante) en utilisant par exemple un modèle avec peu de paramètres ou en contraingnant le modèle. Par exemple, contraindre un modèle pour le rendre plus simple et réduire le risque de surajustement est ce qu’on appelle la **Régularisation**.
    

Il y a principalement trois méthodes de régularisation mais nous ne les aborderons pas dans cet article:

✔ la Régularization L1

✔ la Régularization L2

✔ la Régularization Elastic net

* **Toujours faire une validation croisée (cross validation)** pour évaluer la performance du modèle sur différents sous-ensembles de données. Cela permet d’obtenir une estimation plus fiable de la capacité de généralisation du modèle. Quelques exemples de techniques de cross validation:
    

✔ hold-out cross-validation

✔ k-fold cross-validation

✔ stratified k-fold cross-validation

✔ leave-one-out cross-validation

Notons qu’ils y a pleins d’autres méthodes non mentionnées comme ***l’ajustement des hyperparamètres, l’assemblement (ensembling) de plusieurs modèles (stacking, bagging, boosting), utilisation de earling stopping, ajout de couches de normalisation et de dropout pour les réseaux de neuronnes etc etc…***

Voyons maintenant un principe assez simple qui nous permettra de décider si notre modèle est en Overfitting ou pas.

### **6\. Le principe de simplicité ou Razoir d’Occam**

> *Le rasoir d’Ockham ou rasoir d’Occam est un principe de raisonnement philosophique entrant dans les concepts de rationalisme et de nominalisme. Également appelé principe de simplicité ou principe de parcimonie, il peut se formuler comme suit :*

> **“Pluralitas non est ponenda sine necessitate”** *(les multiples ne doivent pas être utilisés sans nécessité) dont une formulation plus moderne s’apparente à «les hypothèses suffisantes les plus simples doivent être préférées (il faut et il suffit)».*

> [\*Source : Wikipédia-\*Rasoir d’Ockham](https://fr.wikipedia.org/wiki/Rasoir_d%27Ockham)

Certes nous ne sommes pas en philosophie mais c’est un des principes heuristiques fondamentaux en science, sans être pour autant à proprement parler un résultat scientifique qui s’applique également en Machine Learning.

Le rasoir d’Occam dit simplement qu’il ne faut pas essayer de compliquer des choses qui peuvent être résolues de manière beaucoup plus simple. En d’autres termes, les solutions les plus simples sont les plus généralisables.

**En général, chaque fois que notre modèle n’obéit pas au rasoir d’Occam, il s’agit *probablement* d’un cas d’Overfitting.**

![Source — https://kjtradingsystems.com/](https://cdn-images-1.medium.com/max/800/0*4jWAjyeLJRT-oMgn.jpg align="left")

### **Conclusion**

L’overfitting est un problème fréquent en Machine Learning et savoir l’identifier est un crucial pour construire des modèles robustes et qui généralisent bien. Aussi savoir gérer des cas d’overfitting est un “must” pour tout aspirant Data Scientist/ML Engineer.

En fin de compte pour éviter l’overfitting, toujours se rappeler du rasoir d’Occam et se demander ***« Pourquoi chercher compliqué quand plus simple suffit ? ».***

Merci de m’avoir lu jusque là et n’hésite pas à laisser tes remarques et suggestions en commentaire ou à m’envoyer directement un message sur [mon LinkedIn](https://www.linkedin.com/in/anyantudre/).

**Give a clap if you liked et suis moi pour plus de contenu sur la Data Science et le Machine Learning;-)**

R**éférences**

📚 Approaching (Almost) Any Machine Learning Problem — Abhishek Thakur

📚 Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow -Aurélien Géron

📚 Wikipédia — Rasoir d’Ockham

🔗 Lien du dataset: [**Binary Classification with a Bank Churn Dataset**](https://www.kaggle.com/competitions/playground-series-s4e1/data)