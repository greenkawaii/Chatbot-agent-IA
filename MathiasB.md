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

### Point de contrôle 1

python -c "import config; print(config.LLM_MODEL, config.DB_PATH.exists(), 
config.DOCS_DIR.exists(), config.TOP_K)" ==> qwen2.5:7b True True 3
$env:LLM_MODEL = "qwen2.5:3b"; python -c "import config; print(config.LLM_MODEL)" ==> qwen2.5:3b
Remove-Item Env:LLM_MODEL ==> "no error"

### Questions de réflexion

1. **Pourquoi `BASE_DIR` est-il calculé à partir de `__file__` plutôt qu'à partir du dossier courant ? Que se passerait-il si vous lanciez un script depuis un autre dossier ?**

   `__file__` désigne le fichier `config.py` lui-même. En calculant `BASE_DIR` à partir de son emplacement, les chemins comme `data/ecommerce.db` restent corrects quel que soit le dossier depuis lequel on lance le script. Si on utilisait le dossier courant (`Path.cwd()`), ces chemins seraient résolus depuis le dossier du terminal : le programme pourrait ne pas trouver la base ou les documents, ou chercher au mauvais endroit.

2. **Pourquoi convertir avec `float()` ou `int()` les valeurs lues par `os.getenv` ?**

   Les variables d'environnement sont lues sous forme de chaînes de caractères. La conversion permet d'obtenir le type attendu pour les calculs et les appels : `float()` pour la température, qui peut avoir une valeur décimale, et `int()` pour les délais ou les nombres d'étapes. Sans conversion, par exemple, le délai `"120"` resterait du texte au lieu d'un entier.

3. **Pourquoi utiliser une température basse (`0.1`) pour un agent qui écrit du SQL ? Dans quel autre usage d'un LLM préféreriez-vous une température plus élevée ?**

   Une température basse rend les réponses plus prévisibles et limite les variations. C'est préférable pour générer du SQL, où l'on veut des requêtes cohérentes et exactes plutôt que créatives. Une température plus élevée conviendrait davantage à une tâche de création, comme proposer des idées, rédiger une histoire ou varier des formulations.

4. **Quels réglages de ce fichier risquent le plus d'être modifiés pendant la phase de mise au point ? Lesquels ne devraient presque jamais changer ?**

   Pendant la mise au point, on peut ajuster le modèle (`LLM_MODEL`), la température, la taille du contexte, le délai d'attente et les paramètres de recherche RAG (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`), ainsi que les limites de l'agent, selon les résultats observés.

   Les chemins par défaut de la base, des documents et des journaux, ainsi que le nom du modèle d'embeddings, devraient rester stables une fois le projet configuré. Ils ne changent que si l'organisation des fichiers ou les besoins du projet changent.