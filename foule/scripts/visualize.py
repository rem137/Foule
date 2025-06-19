"""Visualize the best Fouloide stored in 'meilleur.pt'."""

import random
import pygame

from ..environment import generer_grille, ajouter_pomme, mise_a_jour_luminosite, get_light
from ..evolution import charger_meilleur

TAILLE_CASE = 16
LARGEUR_GRILLE = 40
HAUTEUR_GRILLE = 40
INFO_PANEL_WIDTH = 300
TAILLE_ECRAN = (LARGEUR_GRILLE * TAILLE_CASE + INFO_PANEL_WIDTH, HAUTEUR_GRILLE * TAILLE_CASE)

FOND = (30, 30, 30)
GRILLE = (50, 50, 50)
POMME = (200, 50, 50)
BLANC = (255, 255, 255)
GRIS_FONCE = (60, 60, 60)


def draw_interface(ecran, fouloide, seed):
    offset_x = LARGEUR_GRILLE * TAILLE_CASE + 10
    font = pygame.font.SysFont("Arial", 18)
    infos = [
        f"Pommes : {fouloide.nb_pommes_mangees}",
        f"Faim : {fouloide.faim:.2f}",
        f"Ticks : {fouloide.ticks}",
        f"Seed : {seed}"
    ]
    for i, text in enumerate(infos):
        y = 20 + i * 30
        label = font.render(text, True, BLANC)
        ecran.blit(label, (offset_x, y))


def visualize():
    pygame.init()
    ecran = pygame.display.set_mode(TAILLE_ECRAN)
    pygame.display.set_caption("Visualisation du Meilleur Fouloïde")
    clock = pygame.time.Clock()
    fouloide = charger_meilleur()
    seed = random.randint(0, 10000)
    py_rng = random.Random(seed)
    rng = random.Random(seed)
    grille = generer_grille(None, py_rng)
    fouloide.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        fouloide.nb_pommes = ajouter_pomme(grille, fouloide.nb_pommes, 30, None, py_rng)
        mise_a_jour_luminosite(grille)
        fouloide.agir(grille)

        ecran.fill(FOND)
        for y in range(HAUTEUR_GRILLE):
            for x in range(LARGEUR_GRILLE):
                case = grille["cases"][y][x]
                rect = pygame.Rect(x * TAILLE_CASE, y * TAILLE_CASE, TAILLE_CASE, TAILLE_CASE)
                if isinstance(case, dict):
                    l = max(0.0, min(1.0, get_light(grille["temps"], case)))
                    color = (int(30 + l * 100), int(30 + l * 100), int(30 + l * 100))
                else:
                    color = GRILLE
                pygame.draw.rect(ecran, color, rect)
                if case.get("type") == "pomme":
                    pygame.draw.circle(ecran, POMME, rect.center, TAILLE_CASE // 3)
                pygame.draw.rect(ecran, GRILLE, rect, 1)

        pygame.draw.rect(ecran, (150, 50, 50) if fouloide.est_mort else (100, 200, 250),
                         (fouloide.x * TAILLE_CASE, fouloide.y * TAILLE_CASE, TAILLE_CASE, TAILLE_CASE))
        draw_interface(ecran, fouloide, seed)
        pygame.display.flip()
        clock.tick(10)
    pygame.quit()


if __name__ == "__main__":
    visualize()
