## Étape 0 - Préparer son poste et son projet

J'ai perdu pas mal de temps au début sur l'initialisation, j'avais commencé sur py 3.14, je m'etais dis que ça n'aurrait aucune incidence sur la suite, mais en voyan scikit-klearn dans le requirement txt j'ai compris qu'il fallait une version stable python, compatible. Donc j'ai du installer une version 3.12.10 supplementaire et refaire les installations
### Point de controle 0

python --version ==> 3.12.10 <br>
python -c "import gradio, faiss, sentence_transformers; print('OK')" ==> OK <br>
python data/create_db.py ==> La base existe déjà. <br>
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
config.DOCS_DIR.exists(), config.TOP_K)" ==> qwen2.5:7b True True 3 <br>
$env:LLM_MODEL = "qwen2.5:3b"; python -c "import config; print(config.LLM_MODEL)" ==> qwen2.5:3b <br>
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
- réponse : en anglais mais différent des rep fr (surement pas de lecture des fichiers interne en fr)
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

   Attention : dans l’implémentation actuelle, `max_retries = 3` limite à **trois appels au total** (l’appel initial est inclus), donc il n’y a que deux pauses avant l’erreur finale : 1 et 2 secondes, soit 3 secondes sans jitter. Le jitter peut ajouter jusqu’à 1 seconde par pause : au maximum, les pauses totalisent donc entre **3 à 5 secondes**.

   Pour estimer le temps total dans le pire cas, il faut ajouter le temps maximal des trois appels : avec un `timeout` de 120 secondes, cela donne environ `3 × 120 + 5 = 365` secondes.

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

   Pour rédiger un rapport de 30 pages, je combinerais les trois approches. Je préparerais d'abord un plan détaillé et le conserverais comme contexte de référence. À chaque échange, j'indiquerais explicitement la partie en cours et les consignes utiles ; le modèle n'aurait ainsi pas à reprendre tout l'historique pour savoir où continuer. Je garderais une fenêtre glissante pour les échanges récents et un résumé mis à jour des décisions, des éléments déjà rédigés et des points restant à traiter. Enfin, un budget de tokens permettrait de vérifier que le plan, le résumé et les échanges utiles tiennent dans la limite du modèle.

   Le plan peut être fourni dans le message système s'il s'agit d'une consigne stable, mais il n'est pas nécessaire d'y mettre l'étape courante : celle-ci peut être transmise dans le message utilisateur avec la demande du moment. Le message système est renvoyé à chaque appel et consomme donc des tokens ; il faut garder ce contexte concis. Pour éviter les oublis ou les répétitions, il est aussi utile de conserver le plan et l'avancement en dehors de la conversation, puis de vérifier le rapport complet à la fin.

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

## Etape 6 - L'outil SQL sécurisé

L'essai avec la saisie `VIP' OR '1'='1` donne **500** avec la f-string : la valeur saisie change la condition SQL et la rend vraie pour tous les clients. Avec le paramètre `?`, le résultat est **0** : la saisie est traitée comme une valeur littérale et non comme du code SQL.

L'outil SQL accepte les requêtes de lecture, ouvre la base en lecture seule et limite le nombre de lignes renvoyées. Pour le test demandé, le regroupement par segment a renvoyé :

| segment | nb |
|---|---:|
| Premium | 90 |
| Standard | 371 |
| VIP | 39 |

La requête `DELETE FROM clients` a été refusée avant exécution. La requête `SELECT montant FROM commandes` a renvoyé `ERREUR SQL : no such column: montant...` : le message est fourni comme texte au lieu de faire planter l'appelant.

### Point de contrôle 6

- `pytest tests/test_etape06_sql.py -v` : **18 tests passés**.
- La base est protégée en lecture seule ; une tentative d'écriture via la connexion est refusée par SQLite.
- Le résultat d'une requête trop longue est limité et annoncé comme tronqué.
- `get_schema()` renvoie les définitions des cinq tables.

### Questions de réflexion

1. **Laquelle des quatre barrières garderiez-vous si vous ne pouviez en garder qu’une ? Pourquoi les autres restent-elles utiles ?**

   Je garderais l'ouverture SQLite en lecture seule (`mode=ro` et `query_only`), car le moteur de base de données refuse alors les écritures même si une requête dangereuse passe par erreur la validation. Les autres barrières restent utiles : la validation refuse tôt les requêtes interdites avec un message clair, les paramètres liés empêchent les valeurs fournies par l'utilisateur de devenir du SQL, et la limite de lignes évite de charger et d'envoyer un résultat trop volumineux.

2. **Expliquez avec vos mots pourquoi la requête avec la f-string renvoie 500 clients.**

   C'est une astuce de "hacking" classique SQl (injection SQL). La condition devient `segment = 'VIP' OR '1'='1'`. Comme `'1'='1'` est toujours vrai, la requête compte toutes les lignes de `clients`, soit 500. Avec `?`, la saisie complète est comparée comme une seule valeur de segment, qui ne correspond à aucun client.

3. **Pourquoi la fonction `run` ne doit-elle jamais lever d’exception ? Qui lira le message d’erreur, et qu’en fera-t-il ?**

   Le message d'erreur doit pouvoir être renvoyé comme observation à l'agent plutôt que d'interrompre le programme. Le LLM qui reçoit cette observation peut comprendre que sa requête est erronée, la corriger et proposer une nouvelle requête. Une requête refusée ne doit toutefois jamais être exécutée pour obtenir cette correction.

4. **Une liste de mots interdits pourrait refuser une requête légitime. Trouvez un exemple.**

   Une validation naïve par expression régulière pourrait refuser `SELECT 'UPDATE' AS exemple`, car le mot interdit apparaît dans une valeur texte, même si la requête ne modifie rien. Il faut distinguer les mots SQL des mots présents dans les littéraux ou commentaires ; la validation de cet outil ignore ces derniers.

5. **Pourquoi présenter le résultat en tableau Markdown plutôt qu’en `print()` d’une liste de tuples Python ?**

   Le tableau Markdown affiche clairement le nom de chaque colonne et l'association entre une valeur et sa colonne. Il est plus facile à lire pour une personne comme pour le LLM qu'une liste de tuples sans présentation ; les nombres sont aussi arrondis de façon uniforme.

6. **En production, sur une base PostgreSQL, quelle protection supplémentaire mettriez-vous en place côté serveur ?**

   Je créerais un compte PostgreSQL dédié à l'agent, avec uniquement le droit `SELECT` sur les tables nécessaires et aucun droit de modification ou d'administration. Les permissions côté serveur restent ainsi une barrière même si l'application a un défaut. Je limiterais également le temps d'exécution des requêtes.

## Etape 7 - Decouper la documentation

Les six documents Markdown sont découpés d'abord par leurs titres `##`, puis par taille si une section est longue. Chaque morceau reçoit le titre du document et celui de sa section pour rester identifiable lorsqu'il sera retrouvé seul. Les fichiers sont chargés en UTF-8 et dans l'ordre alphabétique.

### Point de contrôle 7

- `pytest tests/test_etape07_chunking.py -v` : **3 tests passés**.
- Vérification complémentaire de l'étape 6 : les **18 tests SQL passent** également.
- Avec un chevauchement de 100 caractères, j'obtiens **78 morceaux** pour `chunk_size=200`, **36** pour `chunk_size=500` et **35** pour `chunk_size=1000`.
- Pour `chunk_size=200`, les morceaux sont plus courts et le contexte d'une idée peut être réparti sur plusieurs morceaux ; le préfixe du document et de la section aide à identifier le sujet.

### Questions de réflexion

1. **Testez `load_corpus` avec des tailles de 200, 500 et 1 000 caractères. Combien de morceaux obtenez-vous ? Lisez quelques morceaux de 200 caractères : restent-ils compréhensibles ?**

   J'obtiens respectivement **78**, **36** et **35** morceaux (avec un chevauchement de 100 caractères). Les morceaux de 200 caractères sont plus ciblés, mais une explication longue peut être divisée en plusieurs morceaux. Le titre du document et de la section apporte du contexte ; le chevauchement aide à ne pas perdre complètement une idée coupée à la frontière.

2. **Pourquoi le préfixe `[Titre — Section]` améliore-t-il la recherche ? Donnez un exemple tiré des documents.**

   Le préfixe ajoute le contexte du passage à son contenu. Par exemple, une règle sur le délai de retour peut être identifiée comme appartenant à la section « Délai de rétractation » du document sur la politique des retours, même si le morceau ne répète pas le titre dans son texte.

3. **Que se passerait-il sans chevauchement si une règle importante était coupée en deux ?**

   La première partie pourrait se trouver à la fin d'un morceau et la suite au début du suivant. Une recherche ne retrouvant qu'un seul morceau donnerait une règle incomplète. Le chevauchement augmente les chances qu'un passage utile contienne le contexte nécessaire.

4. **Pourquoi lire les fichiers avec `encoding="utf-8"` est-il indispensable sous Windows ?**

   L'encodage par défaut peut dépendre de la configuration Windows et ne pas être UTF-8. Les accents français risqueraient alors d'être mal décodés ou de provoquer une erreur. Préciser UTF-8 garantit que les caractères des documents sont lus correctement.

5. **Nos documents sont bien structurés en sections `##`. Comment découperiez-vous un PDF de 80 pages sans titres ?**

   Je commencerais par extraire le texte et repérer des frontières sémantiques possibles, comme les pages, les paragraphes ou les changements de sujet. Ensuite, je créerais des morceaux de taille raisonnable avec un chevauchement, en conservant si possible le numéro de page et les titres détectés pour retrouver la source et garder le contexte.

## Etape 8 - Embeddings, FAISS et recherche documentaire

Le modèle `paraphrase-multilingual-MiniLM-L12-v2` produit des vecteurs de **384 dimensions**. Pour les trois phrases de l'expérience, la matrice de similarité cosinus arrondie à deux décimales est :

| | Question sur le retour | Délai de retour de 30 jours | Frais de livraison |
|---|---:|---:|---:|
| Question sur le retour | 1,00 | 0,57 | 0,22 |
| Délai de retour de 30 jours | 0,57 | 1,00 | 0,16 |
| Frais de livraison | 0,22 | 0,16 | 1,00 |

La phrase sur le délai de retour est donc la plus proche de la question. Pour « Quel est le délai de retour pour un client VIP ? », le premier passage retrouvé vient de `02_politique_retours.md`, section « Délai de rétractation », avec un score de 0,79. Les deux autres passages traitent aussi des règles VIP ou du remboursement.

### Point de contrôle 8

- `pytest tests/test_etape08_recherche.py -v` : **2 tests passés** avec le faux embedder.
- Vérification des étapes 6 et 7 : leurs **21 tests** passent également.
- Avec le vrai embedder et le jeu de huit questions, le hit rate mesuré est :

| Taille des morceaux | Hit rate @1 | Hit rate @3 |
|---:|---:|---:|
| 300 | 100 % | 100 % |
| 500 | 100 % | 100 % |
| 800 | 100 % | 100 % |

### Questions de réflexion

1. **Pourquoi faut-il vectoriser la question avec le même modèle que les documents ?**

   Le modèle place les textes dans un même espace vectoriel. Si les documents et la question étaient encodés par des modèles différents, leurs vecteurs ne seraient pas comparables de façon fiable : une proximité numérique ne représenterait plus nécessairement une proximité de sens.

2. **Pourquoi normaliser les vecteurs avant de les ranger dans un `IndexFlatIP` ? Que mesurerait l’index sinon ?**

   Avec des vecteurs normalisés, le produit scalaire utilisé par `IndexFlatIP` est égal à la similarité cosinus. Sans normalisation, le produit scalaire dépend aussi de la longueur des vecteurs ; l'index mesurerait alors une combinaison de direction et de norme plutôt que seulement la proximité sémantique voulue.

3. **La recherche renvoie toujours k passages, même pour « Quel est le salaire moyen des employés ? ». Quel risque pour l’agent ? Proposez deux parades.**

   L'agent pourrait prendre des passages sans rapport pour des preuves et inventer une réponse à une question que la documentation ne couvre pas. Deux parades sont d'utiliser un seuil minimal de score pour écarter les passages trop éloignés, et d'indiquer dans le prompt qu'il doit répondre que l'information n'est pas disponible quand les observations ne la documentent pas.

4. **Comparez votre hit rate avec `chunk_size = 300`, `500` et `800`. Que concluez-vous ? Votre jeu de 8 questions suffit-il pour conclure solidement ?**

   Les trois tailles obtiennent ici 100 % en hit rate @1 et @3. Sur ce petit jeu, je ne vois donc pas de différence mesurable entre elles. Huit questions ne suffisent pas pour conclure solidement : il faudrait un jeu plus grand et varié, représentatif des demandes réelles, avec davantage de formulations et de questions difficiles.

5. **Pourquoi le RAG est-il préférable au fine-tuning pour une politique de retours qui change chaque année ? Et préférable à coller toute la documentation dans le prompt ?**

   Avec le RAG, on met à jour les documents indexés sans réentraîner le modèle, ce qui facilite la prise en compte d'une nouvelle politique. Contrairement à l'envoi de toute la documentation à chaque question, la recherche ne transmet que quelques passages pertinents : le prompt est plus court et les éléments fournis sont ciblés.

6. **FAISS ne stocke que des vecteurs. Citez une solution qui stocke aussi les textes et les métadonnées, et dites quand vous la préféreriez.**

   Chroma peut stocker les vecteurs avec les textes et des métadonnées. Je le préférerais pour un corpus persistant ou plus volumineux, quand je veux gérer plus simplement l'ajout, la mise à jour et la récupération des documents et de leurs métadonnées.

## Etape 9 - Lire la décision du LLM et router

`parse_llm_output` accepte un objet JSON seul ou entouré de texte et de balises Markdown. Il normalise l'action en minuscules, accepte `action_input` si `input` manque et conserve les paramètres éventuels. `route_query` transforme une décision finale en réponse, appelle l'outil correspondant ou renvoie une erreur lisible si le JSON est invalide ou si l'outil n'existe pas.

### Point de contrôle 9

- `pytest tests/test_etape09_routing.py -v` : **8 tests passés**.
- Avec le vrai `SQLTool`, la requête `SELECT COUNT(*) AS n FROM produits` renvoie un `RouteResult` de type `tool` dont le résultat contient **30**.
- Une action `envoyer_email` inconnue renvoie une erreur indiquant « Outil inconnu » et les outils valides.
- Vérification complémentaire : les étapes 3, 4, 6, 7 et 8 passent également, soit **28 tests**.

### Questions de réflexion

1. **Pourquoi l’expression `\{.*\}` doit-elle être « gourmande » (`.*` et non `.*?`) ? Testez les deux avec un JSON qui contient un objet imbriqué.**

   Avec un objet imbriqué comme `{"action":"final_answer","input":{"texte":"ok"}}`, l'expression gourmande va du premier `{` au dernier `}` et capture l'objet complet. L'expression non gourmande s'arrête au premier `}`, celui de l'objet intérieur, et ne capture qu'un fragment JSON incomplet. Si le texte contient plusieurs objets JSON séparés, une expression gourmande peut au contraire les capturer ensemble et échouer ; le parseur est donc conçu pour le cas attendu d'un seul objet dans la réponse.

2. **Pourquoi renvoyer un `RouteResult` d’erreur plutôt que de lever une exception quand l’outil est inconnu ?**

   Une erreur sous forme de résultat peut être ajoutée à la conversation comme observation. Le LLM peut alors corriger son action au prochain tour, sans interrompre toute la boucle ni faire planter l'application.

3. **Que faut-il modifier, et où, pour ajouter un troisième outil à l’agent ?**

   Il faut d'abord créer l'outil avec une méthode `run`, puis l'ajouter au dictionnaire `tools` passé au routeur (et, plus tard, à l'agent). Il faut également autoriser et documenter cette action dans le system prompt. Le routeur peut rester générique tant que l'outil respecte l'interface attendue.

4. **Le dictionnaire `tools` accepte « n’importe quel objet qui a une méthode `run` ». Quel avantage pour les tests ?**

   Les tests peuvent fournir un petit faux outil qui enregistre ses arguments et renvoie un résultat connu. Ils vérifient le routage sans devoir initialiser la vraie base de données, le modèle ou le service externe.

5. **Ce mécanisme ressemble au tool calling natif proposé par les API de LLM. Qu’est-ce que le natif apporterait, et qu’est-ce que notre version « manuelle » vous a permis de comprendre ?**

   Le tool calling natif fournit une structure d'appel d'outil directement dans la réponse de l'API, ce qui évite souvent de demander au modèle de fabriquer lui-même du JSON dans du texte. La version manuelle rend visibles les étapes essentielles : lire une décision, valider l'action, appeler uniquement un outil autorisé et renvoyer son résultat au modèle. Dans les deux cas, le code doit garder le contrôle et vérifier les appels avant leur exécution.

## Etape 10 - La boucle de l'agent

La classe `Agent` relie le client LLM, le routeur et les outils. Pour chaque question, elle ajoute celle-ci à la mémoire de conversation, puis crée un brouillon avec le system prompt et l'historique conservé. À chaque étape, le LLM produit une décision JSON que `route_query` interprète. Si un outil est appelé, son résultat est ajouté au brouillon sous la forme `OBSERVATION (<outil>)`, avec la décision brute du modèle juste avant. Une réponse finale conclut la boucle.

Le brouillon est propre à une question : les observations et les décisions intermédiaires ne sont pas ajoutées à la mémoire longue. Seuls la question et la réponse finale y sont conservées. Si le nombre maximal d'étapes est atteint, l'agent demande au LLM une réponse finale lors d'un dernier appel. Si le client LLM lève `LLMError`, l'agent renvoie un message indiquant que le service est indisponible. Le mode console affiche les étapes, accepte `reset` pour vider la mémoire et `quit` pour quitter.

### Point de contrôle 10

- `pytest tests/test_etape10_agent.py -v` : **5 tests passés** avec un faux LLM.
- Vérification conjointe de `tests/test_etape09_routing.py` et `tests/test_etape10_agent.py` : **13 tests passés**.
- Les tests couvrent notamment l'appel d'un outil suivi d'une réponse finale, l'ajout de son observation au brouillon, la correction après une réponse mal formée, la demande de réponse finale après la limite d'étapes, la mémoire conversationnelle, la réinitialisation et le message renvoyé quand le LLM est indisponible.
- Le mode console avec Ollama réel n'a pas encore été vérifié. Les tests avec le faux LLM valident la boucle et ses garde-fous, mais ne prouvent pas que le modèle réel choisira les bons outils ou calculera les bons résultats.

### Questions de réflexion

1. **Pourquoi limiter le nombre d'étapes ? Que se passerait-il sans cette limite si le LLM répétait la même requête erronée ?**

   La limite empêche une boucle infinie et borne le nombre d'appels au LLM et aux outils. Sans elle, si le modèle répétait une requête erronée ou une action qui ne résout pas le problème, l'agent pourrait continuer indéfiniment, perdre du temps et consommer des ressources. Ici, après la limite, l'agent demande une réponse finale ; si elle n'est toujours pas exploitable comme réponse finale, il renvoie un message d'excuse.

2. **Expliquez la différence entre la mémoire longue et le brouillon. Que se passerait-il si toutes les observations restaient en mémoire ?**

   La mémoire longue conserve les questions de l'utilisateur et les réponses finales entre les tours. Le brouillon ne sert qu'à traiter la question courante : il contient les décisions du LLM et les observations des outils, puis est abandonné. Si toutes les observations restaient dans l'historique, celui-ci grossirait rapidement, pourrait dépasser la fenêtre de contexte et mélanger les résultats d'anciennes questions avec la demande actuelle.

3. **Le message d'observation est envoyé avec le rôle `user`. Pourquoi pas `assistant` ? Pourquoi pas `system` ?**

   Dans le format de conversation utilisé ici, le rôle `user` sert à transmettre au modèle les informations reçues de l'extérieur, notamment le résultat d'un outil. Le rôle `assistant` est réservé aux messages produits par le modèle, par exemple sa décision JSON précédente. Le rôle `system` porte les consignes stables et privilégiées de l'agent ; une observation ponctuelle ne doit pas être présentée comme une consigne système.

4. **Pour « panier moyen 2025 selon la définition officielle », votre agent a-t-il consulté la documentation avant d'écrire le SQL ? Sinon, que changeriez-vous dans votre prompt ?**

   Ce point doit être vérifié avec une exécution réelle et en lisant la trace des étapes ou `logs/llm_exchanges.jsonl` : les tests avec un faux LLM ne permettent pas de savoir quelle séquence le modèle réel choisira. Le system prompt demande déjà d'appeler `search_docs` en cas de doute sur la définition d'un indicateur et donne un exemple où la documentation est consultée avant le SQL. Si la trace montre que le modèle passe directement à la requête, je renforcerais cette consigne en exigeant explicitement une recherche documentaire pour le panier moyen et en précisant de ne calculer qu'après avoir reçu la définition.

5. **Pour « Et combien d'entre eux sont VIP ? », qu'est-ce qui permet à l'agent de comprendre « eux » ? Vérifiez la réponse avec votre propre requête SQL.**

   « Eux » renvoie aux clients actifs mentionnés dans la question précédente. L'agent peut faire ce lien parce que la mémoire longue conserve les derniers échanges et que `Conversation.messages()` les renvoie au LLM avec la nouvelle question. Le résultat attendu indiqué dans le sujet est de 445 clients actifs au 31 décembre 2025 ; le nombre de VIP parmi eux doit être vérifié sur la base, par une requête SQL qui reprend la même définition d'« actif » et filtre ce sous-ensemble sur le segment VIP. Je ne reporte pas de nombre pour ce second calcul sans l'avoir exécuté.

6. **Les tests de l'étape passent avec un faux LLM. Que prouvent-ils, et que ne prouvent-ils pas ?**

   Ils prouvent que, pour les réponses pré-écrites du faux LLM, l'agent enchaîne correctement les étapes, transmet les observations, conserve uniquement les échanges finaux en mémoire, respecte la limite d'étapes et gère une `LLMError`. Ils ne prouvent ni que le vrai modèle répondra toujours en JSON valide, ni qu'il choisira le bon outil, formulera un SQL correct ou donnera un résultat métier exact. Ces points nécessitent des essais avec Ollama et une vérification indépendante des résultats, notamment par SQL.

## Etape 11 - Faire le point : tests et journaux

### Point de contrôle 11

- La commande de tests des étapes 3, 4 et 6 à 10 a donné **41 tests passés** en 4,16 secondes.
- Le journal contient **15 appels** : l'appel de test « Réponds juste OK. » et **14 appels de l'agent réel** pour les six questions de l'étape 10.
- Sur les 14 appels de l'agent, la durée moyenne est de **2,48 s** et le nombre moyen de tokens est de **2 206 en entrée** et **75 en sortie**. La première question a nécessité deux appels en 21,82 s ; les demandes à trois étapes ont pris 3,55 s pour le chiffre d'affaires et 3,45 s pour le panier moyen. Ces temps ne sont qu'une mesure de cette session : un appel initial a duré 21,39 s, ce qui augmente la moyenne.
- Les réponses observées sont correctes pour le nombre de clients VIP (**39**), le chiffre d'affaires net 2025 (**166 091,86 €**) et le délai de retour VIP (**60 jours**). Deux résultats sont erronés : l'agent donne **61,84 €** au lieu de **115,99 €** pour le panier moyen et **461** au lieu de **445** pour les clients actifs.
- Analyse de l'erreur sur le panier moyen : la requête divise la somme des lignes par `COUNT(c.commande_id)` après une jointure avec `lignes_commande`. Une commande ayant plusieurs lignes est comptée plusieurs fois. Il faut compter les commandes distinctes (`COUNT(DISTINCT c.commande_id)`).
- Analyse de l'erreur sur les clients actifs : la requête compte les clients ayant passé n'importe quelle commande en 2025. Elle ne filtre ni les commandes livrées ni la fenêtre des 365 jours précédant le 31 décembre 2025. Avec la définition du glossaire, la requête de référence donne **445**.
- Pour la question de suivi sur les VIP parmi ces clients actifs, la réponse (**39**) correspond au décompte vérifié, mais la requête du LLM compte les clients VIP inscrits avant la date de référence, au lieu de joindre le sous-ensemble des clients actifs. Le résultat est donc juste par coïncidence, pas grâce à une requête qui exprime correctement la demande.

Pour refaire les vérifications et afficher les statistiques depuis le dossier `datapilot` :

```powershell
pytest tests/test_etape03_client.py tests/test_etape04_memoire.py tests/test_etape06_sql.py tests/test_etape07_chunking.py tests/test_etape08_recherche.py tests/test_etape09_routing.py tests/test_etape10_agent.py -q
python
```

Puis, dans l'interpréteur Python :

```python
import pandas as pd
logs = pd.read_json("logs/llm_exchanges.jsonl", lines=True)
agent_logs = logs[logs["messages"].apply(
    lambda messages: any(message.get("role") == "system" for message in messages)
)]
print(agent_logs[["duration_s", "prompt_tokens", "completion_tokens"]].describe())
print(agent_logs.iloc[-1]["answer"])
```

L'agent a maintenant été lancé avec Ollama et ses appels sont dans `logs/llm_exchanges.jsonl`. Pour reproduire le calcul des statistiques, depuis le dossier `datapilot`, lancer `python` puis exécuter le code pandas ci-dessus. Pour analyser une erreur, comparer la réponse finale et le SQL du LLM avec la définition métier et une requête de référence. Les tests automatisés n'appellent pas le vrai LLM et ne remplissent donc pas ce journal.

### Questions de réflexion

1. **Pourquoi un test qui utiliserait le vrai LLM serait-il « instable » ?**

   Le vrai LLM peut produire des réponses différentes à une même demande, et son comportement dépend du modèle, de ses paramètres, du contexte transmis et de la disponibilité du service. Le résultat d'un test peut donc varier ou échouer pour une raison extérieure à la mécanique testée. Un faux LLM renvoie au contraire des réponses prédéfinies, ce qui rend les tests rapides et reproductibles.

2. **Le nombre de tokens lus (`prompt_tokens`) augmente au fil des étapes d'une même question. Pourquoi ?**

   À chaque nouvel appel, l'agent renvoie au LLM le brouillon de la question en cours : les messages précédents du modèle et les observations des outils s'y ajoutent au fur et à mesure. Le prompt contient donc généralement davantage de texte à traiter à chaque étape, ce qui augmente le nombre de tokens d'entrée.

3. **Quelle est la durée moyenne d'un appel sur votre poste ? Combien de temps l'utilisateur attendra-t-il pour une question qui nécessite trois étapes ? Quelle conséquence pour l'interface ?**

   La durée moyenne mesurée est de **2,48 s**. Cette moyenne est fortement influencée par un appel de 21,39 s ; elle dépend du poste et de l'état du modèle, et mérite d'être mesurée sur d'autres sessions. Dans cette exécution, les questions qui ont nécessité trois appels ont pris 3,55 s et 3,45 s au total. L'attente correspond à la somme des appels LLM et du temps passé dans les outils. L'interface devra signaler que le traitement est en cours, rester réactive et afficher éventuellement la progression entre les étapes.

4. **Un test unitaire vert garantit-il que l'agent donnera le bon chiffre d'affaires ? Justifiez.**

   Non. Les tests unitaires vérifient le comportement mécanique du code avec des entrées contrôlées : parsing, routage, exécution des outils, mémoire et gestion des erreurs. Ils ne garantissent pas que le vrai LLM choisira la bonne démarche ni que sa requête correspondra à la définition métier. Il faut également vérifier la réponse de l'agent sur des scénarios réels et comparer le résultat à une requête SQL de référence.
