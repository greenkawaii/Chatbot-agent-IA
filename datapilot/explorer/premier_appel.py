import sys
import time
from pathlib import Path

import requests


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pour importer config
import config

payload = {
    "model": config.LLM_MODEL,
    "messages": [
        # TODO : un message "user" qui demande, en une phrase, ce qu'est un panier moyen
        # {"role": "system", "content": "Tu réponds uniquement en anglais."}, #test anglais
        #{"role": "user", "content": "Donne le nom et la capitale de trois pays. Réponds en JSON."}, #test capital
        {"role": "user", "content": "Comment je m'appelle ?"}
    ],
    # "format": "json", # Test format
    "stream": False,
    "options": {"temperature": config.LLM_TEMPERATURE},
}
debut = time.perf_counter()
reponse = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=120)
duree = time.perf_counter() - debut
# TODO : afficher le code HTTP, le texte de la réponse, les deux compteurs de tokens et la durée
code_http = reponse.status_code
print(f"Code HTTP = {code_http}")
reponse.raise_for_status()

resultat = reponse.json()
texte_reponse = resultat["message"]["content"]
tokens_entree = resultat["prompt_eval_count"]
tokens_sortie = resultat["eval_count"]

print(f"Réponse = {texte_reponse}")
print(f"Tokens entrée = {tokens_entree}, tokens sortie = {tokens_sortie}")
print(f"Durée = {duree:.2f} s")