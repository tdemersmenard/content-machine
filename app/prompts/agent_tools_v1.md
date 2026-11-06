---
name: agent_tools
version: v1
description: Description de chaque outil de l'agent, une section par outil. Le texte est envoyé tel quel dans la définition des outils.
---
# system
## ingest
Télécharge une vidéo depuis une URL, ou lit un fichier local, puis la transcrit et lit ses images. Renvoie `video_id`. Une vidéo déjà connue ne coûte rien. Sinon, compte environ 0,30 $ et 2 minutes. Mets `is_own_content` à vrai seulement si l'utilisateur dit que c'est sa vidéo.

## analyze
Analyse une vidéo ingérée : hook, angle, structure, raisons du succès, patterns extraits. Renvoie un résumé assez court pour comparer plusieurs vidéos. Une vidéo déjà analysée ne coûte rien. Sinon, compte environ 1,40 $ et 7 minutes. N'utilise `force` que si l'utilisateur demande de refaire l'analyse.

## search_patterns
Recherche sémantique dans la bibliothèque de patterns. Gratuit. Utile pour savoir si un mécanisme est déjà connu.

## top_patterns
Les patterns de cette niche, du mieux noté au moins bien noté. Gratuit. `tested` à faux veut dire que le score de 0,5 n'est qu'un a priori : aucune vidéo publiée ne l'a mesuré.

## get_niche
Fiche avatar, faits vécus du créateur et sujets disponibles dans la banque. Gratuit.

## adapt
Adapte une vidéo déjà analysée à la niche et écrit `n_variantes` scripts. Environ 1,30 $ et 7 minutes par script. Une variante qui répète une idée déjà produite est écartée automatiquement.

## generate
Propose des idées à partir des patterns, des sujets et des leçons, puis écrit `count` scripts. Environ 1,20 $ et 6 minutes par script. Une idée qui répète un script déjà produit est écartée automatiquement. Il peut donc y avoir moins de scripts que demandé : regarde `repeated_ideas_dropped` et `failures`.

## get_performance
Stats d'un script publié : tous ses relevés, son score par rapport à la baseline et ses post-mortems. Gratuit. Sans `script_id`, prend le dernier script qui a des stats. C'est le premier outil à appeler quand on te demande pourquoi une vidéo a marché ou floppé.

## run_post_mortem
Génère le diagnostic d'un script s'il n'existe pas déjà, à partir de 72 h et d'un relevé noté. Environ 0,10 $.

## synthesize_lessons
Relit tous les post-mortems et réécrit les leçons sans attendre le seuil de 5 résultats. Environ 0,05 $. Ne le fais que si l'utilisateur le demande.

## learning_report
Rapport de suivi : vidéos notées, gagnantes et perdantes, meilleurs et pires patterns, précision des prédictions, leçons actives. Gratuit, calculé sans Claude.

# user
Ce fichier ne sert qu'à décrire les outils.
