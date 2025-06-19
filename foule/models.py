"""Neural network models used in the simulation."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn as nn

from . import environment as env


class ResidualBlock(nn.Module):
    """Simple residual convolution block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + x)


class NeuralFouloide(nn.Module):
    """Simple convolutional policy network controlling a Fouloide."""

    def __init__(self, rayon_vision: int = 3) -> None:
        super().__init__()
        self.rayon_vision = rayon_vision
        self.actions: List[str] = ["rien", "haut", "bas", "gauche", "droite", "manger"]
        self.nb_actions = len(self.actions)

        size = 2 * rayon_vision + 1
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(32),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * size * size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.nb_actions),
            nn.Tanh(),
        )

        self.device = torch.device("cpu")
        self.reset()

    def reset(self) -> None:
        self.x = env.LARGEUR_GRILLE // 2
        self.y = env.HAUTEUR_GRILLE - 4
        self.faim = 0.5
        self.dopamine = 0.5
        self.nb_pommes_mangees = 0
        self.ticks = 0
        self.est_mort = False
        self.nb_pommes = 0

    def grille_locale(self, grille):
        taille = 2 * self.rayon_vision + 1
        vision = np.zeros((2, taille, taille), dtype=np.float32)

        gx, gy = self.x, self.y
        cases = grille["cases"]
        temp = grille["temps"]

        for dy in range(-self.rayon_vision, self.rayon_vision + 1):
            for dx in range(-self.rayon_vision, self.rayon_vision + 1):
                ny = gy + dy
                nx = gx + dx
                iy = dy + self.rayon_vision
                ix = dx + self.rayon_vision

                if 0 <= nx < env.LARGEUR_GRILLE and 0 <= ny < env.HAUTEUR_GRILLE:
                    case = cases[ny][nx]
                    vision[0, iy, ix] = 1.0 if case["type"] == "pomme" else 0.0
                    vision[1, iy, ix] = env.get_light(temp, case)

        return torch.tensor(vision, dtype=torch.float32).unsqueeze(0).to(self.device)

    def choisir_actions(self, grille):
        vision_tensor = self.grille_locale(grille)
        with torch.no_grad():
            sortie = self.forward(vision_tensor).squeeze(0).cpu().numpy()
        return [self.actions[i] for i, val in enumerate(sortie) if val > 0.5]

    def agir(self, grille):
        if self.est_mort:
            return

        actions = self.choisir_actions(grille)
        dx, dy = 0, 0

        if "haut" in actions:
            dy -= 1
        if "bas" in actions:
            dy += 1
        if "gauche" in actions:
            dx -= 1
        if "droite" in actions:
            dx += 1

        oldx, oldy = self.x, self.y
        self.x = max(0, min(env.LARGEUR_GRILLE - 1, self.x + dx))
        self.y = max(0, min(env.HAUTEUR_GRILLE - 1, self.y + dy))

        if "manger" in actions:
            case = grille["cases"][self.y][self.x]
            if case["type"] == "pomme":
                case["type"] = None
                self.faim = max(0, self.faim - 0.4)
                self.nb_pommes_mangees += 1
                self.nb_pommes -= 1

        lumi = env.get_light(grille["temps"], grille["cases"][self.y][self.x])
        if lumi < 0.3:
            self.faim = np.clip(self.faim + (0.02 + (0.3 - lumi) * 0.05), 0, 1.0)

        if oldx != self.x or oldy != self.y:
            self.faim = np.clip(self.faim + 0.01 + self.faim * 0.01, 0, 1.0)
        if lumi < 0.5:
            self.faim = np.clip(self.faim + 0.01 + self.faim * 0.01, 0, 1.0)
        if oldx == self.x and oldy == self.y:
            self.faim = np.clip(self.faim + 0.01 + self.faim * 0.001, 0, 1.0)

        self.ticks += 1
        if self.faim >= 1.0:
            self.est_mort = True

    def mutate(self, taux: float) -> None:
        for param in self.parameters():
            with torch.no_grad():
                mutation_mask = torch.rand_like(param) < taux
                param += mutation_mask * torch.randn_like(param) * taux

    def forward(self, vision_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv(vision_tensor)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def crossover(parent1: NeuralFouloide, parent2: NeuralFouloide) -> NeuralFouloide:
    enfant = deepcopy(parent1)
    for p_enf, p1, p2 in zip(enfant.parameters(), parent1.parameters(), parent2.parameters()):
        mask = torch.rand_like(p_enf) < 0.5
        with torch.no_grad():
            p_enf.copy_(torch.where(mask, p1.data, p2.data))
    return enfant
