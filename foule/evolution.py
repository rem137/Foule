"""Evolutionary training loop for NeuralFouloide."""

from __future__ import annotations

import random
import time
from copy import deepcopy
from typing import List
import multiprocessing as mp

import numpy as np
import torch

from . import environment as env
from .models import NeuralFouloide, crossover

# Evolution parameters
TAILLE_POPULATION = 200
NB_GENERATIONS = 10000
NB_TICKS = 2000
TAUX_MUTATION = 0.47


SEEDS = [random.randint(0, 10000) for _ in range(3)]
compteur_seed = 0


def simuler_multi(fouloide: NeuralFouloide, seeds: List[int], n_ticks: int = 300) -> float:
    total_score = 0
    for seed in seeds:
        fouloide = deepcopy(fouloide)
        py_rng = random.Random(seed)
        rng = np.random.RandomState(seed)
        grille = env.generer_grille(rng, py_rng)
        fouloide.reset()
        historique_lumieres: List[float] = []

        while not fouloide.est_mort and fouloide.ticks < n_ticks:
            fouloide.nb_pommes = env.ajouter_pomme(grille, fouloide.nb_pommes, 30, rng, py_rng)
            env.mise_a_jour_luminosite(grille)
            fouloide.agir(grille)
            lumi = env.get_light(grille["temps"], grille["cases"][fouloide.y][fouloide.x])
            historique_lumieres.append(lumi)

        total_score += (
            fouloide.nb_pommes_mangees * 10
            + fouloide.ticks
            + sum(1 if l >= 0.5 else (0.5 if l >= 0.3 else -1) for l in historique_lumieres)
            + (50 if not fouloide.est_mort else 0)
        )

    return total_score / len(seeds)


def charger_meilleur(path: str = "meilleur.pt") -> NeuralFouloide | None:
    try:
        model = NeuralFouloide()
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        return model
    except FileNotFoundError:
        print("[!] Aucun fichier meilleur.pt trouvé.")
        return None


def eval_population(population: List[NeuralFouloide], seeds: List[int], n_ticks: int) -> List[float]:
    """Evaluate the entire population in parallel."""
    with mp.Pool() as pool:
        args = [(deepcopy(ind), seeds, n_ticks) for ind in population]
        return pool.starmap(simuler_multi, args)


def run_evolution() -> None:
    global SEEDS, compteur_seed
    meilleur = charger_meilleur()
    population: List[NeuralFouloide] = []

    while len(population) < TAILLE_POPULATION:
        if meilleur:
            population.append(deepcopy(meilleur))
        else:
            population.append(NeuralFouloide())

    taux_mutation = TAUX_MUTATION
    historique_scores: List[float] = []
    hall_of_fame: List[tuple[float, NeuralFouloide]] = []
    HALL_OF_FAME_MAX = 5

    for generation in range(NB_GENERATIONS):
        start_gen = time.perf_counter()
        print(f"Génération {generation}")

        if compteur_seed % 10 == 0:
            SEEDS = [random.randint(0, 10000) for _ in range(3)]
        compteur_seed += 1

        scores = eval_population(population, SEEDS, NB_TICKS)
        scores_population = list(zip(scores, population))
        scores_population.sort(reverse=True, key=lambda x: x[0])

        scores_gen = [s for s, _ in scores_population]
        best_score, best = scores_population[0]

        historique_scores.append(best_score)

        if len(hall_of_fame) < HALL_OF_FAME_MAX:
            hall_of_fame.append((best_score, best))
        else:
            worst_score = min(hall_of_fame, key=lambda x: x[0])[0]
            if best_score > worst_score:
                hall_of_fame.remove(min(hall_of_fame, key=lambda x: x[0]))
                hall_of_fame.append((best_score, deepcopy(best)))

        if len(historique_scores) > 5:
            historique_scores.pop(0)

        moyenne_prec = np.mean(historique_scores[:-1]) if len(historique_scores) > 1 else best_score
        variation = (best_score - moyenne_prec) / moyenne_prec if moyenne_prec > 0 else 0
        variation = np.clip(variation, -0.5, 0.5)

        MUTATION_BASE = 0.2
        MUTATION_MIN = 0.05
        MUTATION_MAX = 0.6
        STAGNATION_SEUIL = 0.01
        NB_STAGNATION_MAX = 5

        if 'nb_stagnantes' not in locals():
            nb_stagnantes = 0

        if abs(variation) < STAGNATION_SEUIL:
            nb_stagnantes += 1
        else:
            nb_stagnantes = 0

        if nb_stagnantes >= NB_STAGNATION_MAX:
            taux_mutation = min(taux_mutation + 0.05, MUTATION_MAX)
        elif variation > 0.05:
            taux_mutation = max(taux_mutation - 0.02, MUTATION_MIN)
        else:
            taux_mutation += (MUTATION_BASE - taux_mutation) * 0.1

        taux_mutation = np.clip(taux_mutation, MUTATION_MIN, MUTATION_MAX)

        hof_individus = [x[1] for x in hall_of_fame]
        next_population: List[NeuralFouloide] = [deepcopy(x[1]) for x in hall_of_fame]
        next_population.append(deepcopy(best))

        while len(next_population) < TAILLE_POPULATION:
            if len(hof_individus) >= 2:
                p1, p2 = random.sample(hof_individus, 2)
            else:
                # If the hall of fame is still too small, breed the best
                # individual with itself. This avoids ``random.sample``
                # raising ``ValueError`` when the population is < 2.
                p1 = p2 = hof_individus[0]

            enfant = crossover(p1, p2)
            enfant.mutate(taux_mutation)
            next_population.append(enfant)

        population = next_population
        duration = time.perf_counter() - start_gen

        torch.save(best.state_dict(), "meilleur.pt")
        nb_params = sum(p.numel() for p in best.parameters())
        print(
            f">>> Génération {generation} - meilleur {best_score:.2f} | mutation {taux_mutation:.4f} | {duration:.2f}s | params {nb_params}"
        )

