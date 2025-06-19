import pygame
import numpy as np
import random
import sys
from generation import NeuralFouloide, generer_grille, ajouter_pomme, maj_pommes, charger_meilleur, mise_a_jour_luminosite, get_light


# --- Paramètres ---
TAILLE_CASE = 16
LARGEUR_GRILLE = 40
HAUTEUR_GRILLE = 40
INFO_PANEL_WIDTH = 300
TAILLE_ECRAN = (LARGEUR_GRILLE * TAILLE_CASE + INFO_PANEL_WIDTH, HAUTEUR_GRILLE * TAILLE_CASE)

# --- Couleurs ---
FOND = (30, 30, 30)
GRILLE = (50, 50, 50)
POMME = (200, 50, 50)
FOULOIDE = (100, 200, 250)
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

        # juste après la ligne « Faim »
        if i == 1:
            bar_y = y + 20          # petit espace sous le texte
            bar_w = 100
            hunger = max(0, min(1, fouloide.faim))  # normalisé 0-1
            pygame.draw.rect(ecran, GRIS_FONCE, (offset_x, bar_y, bar_w, 10))
            pygame.draw.rect(ecran, (100, 250, 100), (offset_x, bar_y, hunger * bar_w, 10))




def draw_vision(ecran, fouloide):
    vision = fouloide.grille_locale(grid)[1:]  # enlève faim
    taille = fouloide.rayon_vision * 2 + 1
    case_size = 32
    origin_x = LARGEUR_GRILLE * TAILLE_CASE + 20
    origin_y = 150

    # Première passe : Vision des pommes
    for i in range(taille):
        for j in range(taille):
            idx = (i * taille + j) * 2  # *2 car deux valeurs (pomme, lumière)
            pomme_value = vision[idx]
            x = origin_x + j * case_size
            y = origin_y + i * case_size
            rect = pygame.Rect(x, y, case_size, case_size)

            if i == fouloide.rayon_vision and j == fouloide.rayon_vision:
                val = max(0.0, min(1.0, fouloide.getOutputs()[0]))
                color = (150, 50, 50) if fouloide.est_mort else (
                    int(255 * val), int(255 * (1 - val)), 100
                )
            elif pomme_value == -1.0:
                color = GRIS_FONCE
            elif pomme_value > 0.5:
                color = POMME
            else:
                color = GRILLE

            pygame.draw.rect(ecran, color, rect)
            pygame.draw.rect(ecran, BLANC, rect, 1)

    # Deuxième passe : Vision de la luminosité (juste en dessous)
    offset_y = origin_y + (taille + 1) * case_size  # un peu d'espace
    for i in range(taille):
        for j in range(taille):
            idx = (i * taille + j) * 2 + 1  # +1 pour la lumière
            lumi_value = vision[idx]
            x = origin_x + j * case_size
            y = offset_y + i * case_size
            rect = pygame.Rect(x, y, case_size, case_size)

            if lumi_value == -1.0:
                color = GRIS_FONCE
            else:
                l = max(0.0, min(1.0, lumi_value))
                color = (int(255 * l), int(255 * l), int(255 * l))  # Gris clair selon lumière

            pygame.draw.rect(ecran, color, rect)
            pygame.draw.rect(ecran, BLANC, rect, 1)




pygame.init()
font = pygame.font.SysFont("Arial", 18)
ecran = pygame.display.set_mode(TAILLE_ECRAN)
pygame.display.set_caption("Visualisation du Meilleur Fouloïde")
clock = pygame.time.Clock()

# Chargement du meilleur
fouloide = charger_meilleur()

seed = random.randint(0, 10000)
py_rng = random.Random(seed)
rng = np.random.RandomState(seed)
grid = generer_grille(rng, py_rng)
fouloide.reset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    fouloide.nb_pommes = ajouter_pomme(grid, fouloide.nb_pommes, 30, rng, py_rng)
    #maj_pommes(grid)
    mise_a_jour_luminosite(grid)
    fouloide.agir(grid)

    ecran.fill(FOND)
    for y in range(HAUTEUR_GRILLE):
        for x in range(LARGEUR_GRILLE):
            case = grid["cases"][y][x]
            rect = pygame.Rect(x * TAILLE_CASE, y * TAILLE_CASE, TAILLE_CASE, TAILLE_CASE)
            
            # Calcul de la couleur en fonction de la luminosité
            if isinstance(case, dict):
                l = max(0.0, min(1.0, get_light(grid["temps"], case)))
                color = (int(30 + l * 100), int(30 + l * 100), int(30 + l * 100))  # Gris plus ou moins clair
            else:
                color = GRILLE
            
            pygame.draw.rect(ecran, color, rect)

            if case.get("type") == "pomme":
                pygame.draw.circle(ecran, POMME, rect.center, TAILLE_CASE // 3)

            pygame.draw.rect(ecran, GRILLE, rect, 1)  # Bordure




    val = max(0.0, min(1.0, fouloide.getOutputs()[0]))  # sécurité
    couleur_fouloide = (150, 50, 50) if fouloide.est_mort else (
        int(255 * val), int(255 * (1 - val)), 100
    )
    pygame.draw.rect(ecran, couleur_fouloide, (fouloide.x * TAILLE_CASE, fouloide.y * TAILLE_CASE, TAILLE_CASE, TAILLE_CASE))
    draw_interface(ecran, fouloide, seed)
    draw_vision(ecran, fouloide)
    pygame.display.flip()
    clock.tick(10)

pygame.quit()
