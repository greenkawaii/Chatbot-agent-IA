## Étape 0 - Préparer son poste et son projet

### Point de controle 0

python --version ==> 3.12.10
python -c "import gradio, faiss, sentence_transformers; print('OK')" ==> OK
python data/create_db.py ==> La base existe déjà.
Invoke-RestMethod http://localhost:11434/api/tags ==> {@{name=qwen2.5:7b; model=qwen2.5:7b; ....}}

### Questions de réflexion

1. **Pourquoi faire tourner le LLM en local plutôt que d'envoyer les questions à un service en ligne, dans le cas de ShopNova ? Citez un avantage et un inconvénient.**

   Avantage :
   - les questions et les données de ShopNova restent sur les machines de l'entreprise : elles ne sont pas transmises à un service externe. 
   - possibilité de personnaliser le modèle pour l'entreprise, par exemple en l'entraînant sur des données internes.
   
   Inconvénient:
   - Il faut une machine suffisamment puissante pour exécuter le modèle. Et toujours allumé pour les utilisateurs

2. **Pourquoi le premier appel à `ollama run` est-il nettement plus lent que les suivants ?**

   Une hypothèse est que le premier appel doit charger le modèle depuis le disque vers la mémoire (RAM ou mémoire vidéo) et initialiser son exécution. Les appels suivants sont plus rapides tant que le modèle reste chargé en mémoire.

3. **À quoi sert un environnement virtuel ? Que se passerait-il si deux projets avaient besoin de deux versions différentes de Gradio ?**

   Un environnement virtuel isole les dépendances d'un projet de celles des autres projets. Chaque projet peut installer et utiliser sa propre version de Gradio.

4. **Quelles commandes faut-il compter dans le chiffre d'affaires de ShopNova ? Pourquoi cette information ne peut-elle pas se trouver dans la base de données ?**

   Selon le glossaire, il faut compter uniquement les commandes au statut **« livrée »**. Le chiffre d'affaires net est la somme des montants des lignes de commande, remises déduites : `quantité × prix_unitaire × (1 − remise_pct / 100)`. Les frais de livraison sont exclus. Les commandes annulées, en cours ou retournées ne sont pas comptées.

   La base contient les données des commandes (par exemple leur statut, les quantités, les prix et les remises), mais la règle qui définit le chiffre d'affaires est une définition métier. Elle doit être documentée dans le glossaire, car elle ne découle pas nécessairement des noms ou des valeurs des colonnes de la base.
   Cela correspond au filtrage de données que le data analyst/scientist doit effectuer pour calculer le chiffre d'affaires net.

## Étape 1 - Centraliser la configuration

