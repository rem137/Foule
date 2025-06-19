"""Environment utilities for the Fouloide simulation."""

import math
import random
from typing import Dict, Any

import numpy as np

# Grid parameters
LARGEUR_GRILLE = 40
HAUTEUR_GRILLE = 40


def generer_grille(rng: np.random.RandomState | None = None,
                   py_rng: random.Random | None = None) -> Dict[str, Any]:
    """Generate the simulation grid with torches."""
    if rng is None:
        rng = np.random.RandomState()
    if py_rng is None:
        py_rng = random.Random()

    grille = {
        "cases": [[{"luminosite": -1, "bonus_torche": 0.0, "type": None}
                   for _ in range(LARGEUR_GRILLE)]
                  for _ in range(HAUTEUR_GRILLE)],
        "temps": 0.0,
        "torches": []
    }

    for _ in range(5):
        tx = py_rng.randint(0, LARGEUR_GRILLE - 1)
        ty = py_rng.randint(0, HAUTEUR_GRILLE - 1)
        grille["cases"][ty][tx]["type"] = "torche"
        grille["torches"].append((tx, ty))

        for dy in range(-6, 7):
            for dx in range(-6, 7):
                nx = tx + dx
                ny = ty + dy
                if 0 <= nx < LARGEUR_GRILLE and 0 <= ny < HAUTEUR_GRILLE:
                    dist = max(1, abs(dx) + abs(dy))
                    gain = 2.0 / dist
                    grille["cases"][ny][nx]["bonus_torche"] += gain
                    grille["cases"][ny][nx]["bonus_torche"] = min(
                        1.0, grille["cases"][ny][nx]["bonus_torche"])
    return grille


def mise_a_jour_luminosite(grille: Dict[str, Any]) -> None:
    grille["temps"] += 0.01
    if grille["temps"] > 2 * math.pi:
        grille["temps"] -= 2 * math.pi


def get_light(temp: float, case: Dict[str, Any]) -> float:
    bonus = case["bonus_torche"]
    lumi = 0.5 + 0.5 * math.cos(temp)
    lumi += bonus
    return min(1.0, lumi)


def ajouter_pomme(grille: Dict[str, Any], nb_pommes: int,
                  max_pommes: int = 30,
                  rng: np.random.RandomState | None = None,
                  py_rng: random.Random | None = None) -> int:
    if nb_pommes >= max_pommes:
        return nb_pommes

    for _ in range(5):
        y = py_rng.randint(0, HAUTEUR_GRILLE - 1)
        x = py_rng.randint(0, LARGEUR_GRILLE - 1)
        if grille["cases"][y][x]["type"] is None:
            grille["cases"][y][x]["type"] = "pomme"
            grille["cases"][y][x]["vie"] = py_rng.randint(60, 1000)
            nb_pommes += 1
    return nb_pommes


def maj_pommes(grille: Dict[str, Any]) -> None:
    for y in range(HAUTEUR_GRILLE):
        for x in range(LARGEUR_GRILLE):
            cell = grille["cases"][y][x]
            if isinstance(cell, dict) and cell["type"] == "pomme":
                cell["vie"] -= 1
                if cell["vie"] <= 0:
                    grille["cases"][y][x]["type"] = None
