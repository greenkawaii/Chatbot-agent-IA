## Étape 0 - Préparer son poste et son projet

J'ai perdu pas mal de temps au début sur l'initialisation, j'avais commencé sur py 3.14, je m'etais dis que ça n'aurrait aucune incidence sur la suite, mais en voyan scikit-klearn dans le requirement txt j'ai compris qu'il fallait une version stable python, compatible. Donc j'ai du installer une version 3.12.10 supplementaire et refaire les installations
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

## Etape 2 - Premier dialogue avec le LLM

### Point de contrôle 2

**lancement consécutif**
|action|premier lancement|deuxieme lancement|
|---|---|---|
|token entrée|37|37|
|token sortie|342|297|
|durée|7.84s|5.39s|

**Experience**
en anglais :
- token : [29, 84]
- réponse : en anglais mais différent des rep fr (surement pas de lecture des fichier interne en fr)
- temps : 4.92s

Json : 
- token : [47, 82]
- réponse : json format
- temps : 4.88 s

Ollama quitter:
- connextionError

### Question de réflexion

1. **Que constate-t-on quand on demande « Je m’appelle Lina », puis « Comment je m’appelle ? » dans un second appel ?**

   Si le second appel n’envoie que la question « Comment je m’appelle ? », le modèle ne sait pas que je m’appelle Lina : il n’a pas accès automatiquement aux appels précédents. Un LLM n’a donc pas de mémoire persistante entre deux appels. Pour qu’il puisse s’appuyer sur l’échange précédent, le programme doit lui renvoyer aussi les messages de la conversation dans la requête.

2. **Quel est l’effet de `"format": "json"` ? Pourquoi cette option est-elle utile ?**

   Cette option demande à Ollama de produire une réponse au format JSON. Dans mon essai, la réponse était bien structurée en JSON. C’est utile pour un programme, car il peut ensuite analyser les champs et leurs valeurs avec un parseur JSON, au lieu de devoir interpréter du texte libre.

3. **Pourquoi `prompt_eval_count` augmente-t-il avec un message `system` ? Quelle est la conséquence d’un system prompt de 1 000 tokens envoyé à chaque question ?**

   Le message `system` fait partie du texte envoyé au modèle : ses tokens sont donc comptés dans `prompt_eval_count`. Un system prompt de 1 000 tokens renvoyé à chaque question ajoute environ 1 000 tokens d’entrée à chaque appel. Cela demande davantage de calcul et peut augmenter la durée de réponse et l’usage de la fenêtre de contexte.

4. **Quelle exception a été levée quand Ollama était arrêté ? Quel message serait préférable pour un utilisateur non technicien ?**

   D’après mes notes, c’était une erreur de connexion (`requests.exceptions.ConnectionError`). Pour un utilisateur non technicien, un message plus clair serait : « Impossible de contacter le service IA. Vérifiez qu’Ollama est lancé, puis réessayez. »

## Etape 3 - Un client LLM robuste

### Point de contrôle 3

- `pytest tests/test_etape03_client.py -v` : **4 tests passés** (`test_chat_ok_et_payload`, `test_retry_sur_erreur_temporaire`, `test_pas_de_retry_sur_404`, `test_serveur_injoignable`).
- Avec Ollama : `is_available()` a renvoyé `True` et le client a répondu `OK` à la demande. Le journal `logs/llm_exchanges.jsonl` contient une ligne pour cet appel.

### Questions de réflexion

1. **Pourquoi ne faut-il pas réessayer une erreur 404 ? Donnez un exemple concret dans notre projet qui produirait cette erreur.**

   Une erreur 404 indique que la ressource demandée n’existe pas ; refaire exactement la même requête ne la fera pas apparaître. Par exemple, si `LLM_MODEL` indique un modèle qui n’est pas installé dans Ollama, le serveur peut répondre « modèle introuvable ». Il faut corriger le nom ou installer le modèle, pas réessayer automatiquement.

2. **Avec `max_retries = 3` et `backoff_factor = 1`, combien de secondes de pause au maximum avant l’erreur finale ? Que faut-il ajouter pour connaître le temps d’attente total dans le pire cas ?**

   Sans jitter, le client effectue trois tentatives et attend 1 seconde puis 2 secondes entre elles, soit **3 secondes de backoff**. Le jitter ajouté peut ajouter jusqu’à 1 seconde à chacune des deux pauses : le total des pauses est donc au plus 5 secondes. Pour estimer le temps total dans le pire cas, il faut aussi compter le temps maximal des trois requêtes (par exemple jusqu’à `timeout` chacune) : avec un timeout de 120 secondes, cela peut atteindre environ `3 × 120 + 5 = 365` secondes.

3. **Pourquoi le message de l’erreur finale doit-il expliquer quoi faire (« ollama serve ») plutôt que d’afficher seulement l’exception technique ?**

   Une exception technique décrit le problème dans des termes parfois difficiles à comprendre. Indiquer une action, comme vérifier qu’Ollama est lancé avec `ollama serve`, aide l’utilisateur à résoudre le problème sans devoir interpréter une trace technique.

4. **Pourquoi le format JSONL est-il plus pratique qu’un unique gros fichier JSON pour un journal qui grossit en continu ?**

   En JSONL, chaque échange est un objet JSON sur sa propre ligne : on peut ajouter une ligne à la fin sans relire ni réécrire tout le fichier. On peut aussi traiter les entrées une par une. Dans un grand tableau JSON, un ajout doit préserver la structure globale et une interruption pendant l’écriture peut rendre le fichier entier invalide.

5. **Le journal contient les questions des utilisateurs. Quelles précautions faudrait-il prendre en production ?**

   Il faudrait limiter les données enregistrées au strict nécessaire, masquer les informations personnelles ou sensibles, restreindre et protéger l’accès aux journaux, fixer une durée de conservation puis supprimer les données à son terme, et informer les utilisateurs conformément aux règles de confidentialité applicables.

### Pour aller plus loin : jitter

Un jitter ajoute un petit délai aléatoire au temps de backoff. Si beaucoup de clients rencontrent une panne en même temps, cela évite qu’ils retentent tous au même instant et surchargent à nouveau le service. Le délai aléatoire répartit les nouvelles tentatives dans le temps.


## Etape 4

La classe `Conversation` conserve séparément le message système et l’historique des échanges. `messages()` remet toujours le message système en premier, puis les derniers messages de la conversation. La fenêtre glissante garde les questions et les réponses récentes, sans dépasser le nombre de tours demandé.

### Point de contrôle 4

- `pytest tests/test_etape04_memoire.py -v` : **1 test passé**.
- Avec `max_turns=2` et 5 tours enregistrés, `messages()` renvoie bien le message système, puis `Q3`, `R3`, `Q4`, `R4`.
- Vérification complémentaire : les 4 tests de l’étape 3 passent aussi.

### Questions de réflexion

1. **Pourquoi multiplier `max_turns` par 2 ? Que se passerait-il si la fenêtre commençait par une réponse assistant sans sa question ?**

   Un tour comprend deux messages : une question `user` et une réponse `assistant`. Il faut donc conserver jusqu’à `2 × max_turns` messages. Si l’historique commençait par une réponse sans sa question, le modèle perdrait le contexte de cette réponse et pourrait mal comprendre la suite de la conversation.

2. **Pourquoi le system prompt ne doit-il jamais sortir de la fenêtre ?**

   Le system prompt contient les consignes et le rôle de l’assistant. Il doit être renvoyé à chaque appel pour que le modèle suive ces consignes, même après avoir supprimé les anciens tours de conversation.

3. **Comparez la fenêtre glissante, le budget de tokens et le résumé des anciens échanges. Laquelle choisiriez-vous pour un assistant qui aide à rédiger un rapport de 30 pages, et pourquoi ?**

   La fenêtre glissante est simple et garde les échanges les plus récents, mais elle oublie les informations anciennes. Un budget de tokens limite directement la taille du contexte envoyé au modèle, mais il faut décider quelles informations retirer quand la limite est atteinte. Un résumé conserve les points importants d’échanges anciens sous une forme plus courte, mais il peut omettre ou déformer des détails.

   Pour rédiger un rapport de 30 pages, je choisirais un résumé des décisions et des informations importantes, combiné à un budget de tokens pour respecter la limite du modèle. Je garderais aussi les échanges récents en fenêtre glissante afin que les dernières demandes restent dans leur contexte.

4. **L’estimation « caractères / 4 » est grossière. Dans quel cas risque-t-elle de beaucoup se tromper ?**

   Le nombre de caractères ne correspond pas toujours au nombre de tokens. L’estimation peut être très différente pour des textes dans une langue moins bien représentée par le tokenizer, des mots rares, des suites de chiffres, du code ou des caractères spéciaux. Pour obtenir un compte exact, il faudrait utiliser le tokenizer du modèle concerné.

## Etape 5 - Le system prompt, cerveau de l'agent

Le prompt est dans `prompts.py` et son schéma est inséré par `build_system_prompt(schema)`. J'ai ajouté des rappels sur les statuts accentués, les dates ISO, l'absence de montant total dans `commandes` et le calcul des montants à partir de `lignes_commande`. Le chiffre d'affaires net demande de vérifier la définition dans les documents avant de calculer. J'ai également précisé le format JSON et les trois actions possibles.

### Point de contrôle 5

J'ai testé trois questions avec le modèle local, sans mode JSON forcé et sans appeler d'outil. Les trois réponses étaient des JSON valides avec les champs `thought`, `action` et `input`, et les actions attendues dans l'ordre :

1. « Combien de clients ont le segment VIP ? » → `query_database`
2. « Quelle est la définition du panier moyen ? » → `search_docs`
3. « Bonjour, que peux-tu faire ? » → `final_answer`

### Questions de réflexion

1. **Pourquoi demander une réponse en JSON plutôt qu’en phrases ? Que devrait faire votre programme avec une réponse comme « Je pense qu’il faudrait regarder dans la base » ?**

   Le JSON donne une structure prévisible que le programme peut analyser pour choisir une action et lui transmettre son entrée. Une phrase libre n'indique pas de façon fiable l'action à exécuter. Le programme doit tenter de valider la réponse et, si elle n'est pas un JSON valide conforme au contrat, ne pas l'exécuter comme une action ; il peut demander une nouvelle réponse au modèle ou afficher une erreur contrôlée.

2. **À quoi sert le champ `thought`, pour le modèle et pour vous ?**

   Il fournit une justification très brève du choix d'action, ce qui aide à rendre la décision explicite et à déboguer le routage. Ce n'est pas une garantie que la décision est correcte et le programme ne doit pas se fier à ce champ pour exécuter une action.

3. **Pourquoi insérer le schéma automatiquement (`{schema}`) plutôt que de le recopier à la main dans le prompt ?**

   L'insertion du schéma évite d'entretenir deux versions manuelles qui pourraient diverger si la base change. Le modèle reçoit ainsi les tables et colonnes réellement disponibles.

4. **Quels pièges de la base avez-vous mis dans les rappels ? Pour chacun, quelle erreur le modèle ferait-il sans ce rappel ?**

   - Les statuts contiennent des accents (`livrée`, `annulée`, `retournée`) : sans rappel, le filtre SQL pourrait ne pas correspondre aux valeurs réelles.
   - Les dates sont du texte au format ISO : sans rappel, le modèle pourrait utiliser un format ou une fonction de filtrage inadaptés.
   - `commandes` ne contient pas de montant total : sans rappel, le modèle pourrait demander une colonne inexistante au lieu de calculer les lignes.
   - Le prix facturé est dans `lignes_commande`, distinct du prix catalogue dans `produits` : sans rappel, le calcul pourrait utiliser le mauvais prix.
   - Le montant d'une ligne tient compte de la quantité et de la remise ; le chiffre d'affaires net ne compte que les commandes livrées et exclut la livraison : sans rappel, le calcul pourrait ignorer les remises, inclure des commandes non livrées ou les frais d'expédition.

5. **Si vous mettez dans vos exemples la question exacte « Combien de clients VIP avons-nous ? » avec sa réponse, en quoi cela fausse-t-il ensuite l’évaluation de votre agent sur cette même question ?**

   Le modèle pourrait reproduire la réponse mémorisée dans l'exemple au lieu de comprendre la question et d'interroger correctement la base. Le test mesurerait alors sa capacité à répéter l'exemple, et non le bon fonctionnement du routage et du calcul sur une question nouvelle.

6. **Le system prompt est-il une garantie ? Que devra faire le code si le modèle ne le respecte pas ?**

   Non : le prompt est une consigne, pas une garantie. Le code doit valider et parser le JSON, vérifier que les champs et l'action sont autorisés, puis contrôler la requête SQL avant de l'exécuter. En cas de réponse invalide, il doit refuser de lancer l'action et gérer l'erreur proprement, éventuellement en demandant une nouvelle réponse au modèle.
