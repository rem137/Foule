import random
import numpy as np
import math
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
plt.ion()  # mode interactif pour affichage en temps réel
from copy import deepcopy
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Paramètres d'évolution ---
TAILLE_POPULATION = 200  # ou n'importe quel nombre fixe
NB_GENERATIONS = 10000
NB_TICKS = 2000
TAUX_MUTATION = 0.47

# --- Grille ---
LARGEUR_GRILLE = 40
HAUTEUR_GRILLE = 40

historique_best = []
historique_moyenne = []


def generer_grille(rng=None, py_rng=None):
    grille = {
        "cases": [[{"luminosite": -1, "bonus_torche": 0.0, "type": None} for _ in range(LARGEUR_GRILLE)] for _ in range(HAUTEUR_GRILLE)],
        "temps": 0.0,
        "torches": []
    }

    # Ajouter 5 torches
    for _ in range(5):  # Jusqu'à 5 essais max
        tx = py_rng.randint(0, LARGEUR_GRILLE - 1)
        ty = py_rng.randint(0, HAUTEUR_GRILLE - 1)
        grille["cases"][ty][tx]["type"] = "torche"
        grille["torches"].append((tx, ty))

        # Effet de la torche
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                nx = tx + dx
                ny = ty + dy
                if 0 <= nx < LARGEUR_GRILLE and 0 <= ny < HAUTEUR_GRILLE:
                    dist = max(1, abs(dx) + abs(dy))
                    gain = 2.0 / dist
                    grille["cases"][ny][nx]["bonus_torche"] += gain
                    grille["cases"][ny][nx]["bonus_torche"] = min(1.0, grille["cases"][ny][nx]["bonus_torche"])

    

    return grille




def mise_a_jour_luminosite(grille):
    grille["temps"] += 0.01
    if grille["temps"] > 2 * math.pi:
        grille["temps"] -= 2 * math.pi

def get_light(temp, case):
    bonus = case["bonus_torche"]

    # Calcul de la lumière ambiante jour/nuit
    lumi = 0.5 + 0.5 * math.cos(temp)

    # Ajout du bonus de torche éventuel
    lumi += bonus

    # Clamp pour ne pas dépasser 1.0
    lumi = min(1.0, lumi)

    return lumi








def ajouter_pomme(grille, nb_pommes, max_pommes=30, rng=None, py_rng=None):
    if nb_pommes >= max_pommes:
        return nb_pommes  # Rien à faire

    # Essaye de poser une pomme sur une case vide au hasard
    for _ in range(5):  # Jusqu'à 5 essais max
        y = py_rng.randint(0, HAUTEUR_GRILLE - 1)
        x = py_rng.randint(0, LARGEUR_GRILLE - 1)
        if grille["cases"][y][x]["type"] is None:
            grille["cases"][y][x]["type"] = "pomme"
            grille["cases"][y][x]["vie"] = py_rng.randint(60, 1000)
            nb_pommes += 1

    return nb_pommes  # Pas trouvé de case vide


def maj_pommes(grille):
    for y in range(HAUTEUR_GRILLE):
        for x in range(LARGEUR_GRILLE):
            cell = grille["cases"][y][x]
            if isinstance(cell, dict) and cell["type"] == "pomme":
                cell["vie"] -= 1
                if cell["vie"] <= 0:
                    grille["cases"][y][x]["type"] = None

class NeuralFouloide(nn.Module):
    def __init__(self, rayon_vision=3):
        super().__init__()
        self.rayon_vision = rayon_vision
        self.actions = ["rien", "haut", "bas", "gauche", "droite", "manger"]
        self.nb_actions = len(self.actions)

        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),  # entrée: 2 canaux (pomme, lumière)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * (2 * rayon_vision + 1) ** 2, 32),
            nn.ReLU(),
            nn.Linear(32, self.nb_actions),
            nn.Tanh()
        )

        self.device = torch.device("cpu")  # ou "cuda" si GPU

        # Position et état
        self.reset()

    def reset(self):
        self.x = LARGEUR_GRILLE // 2
        self.y = HAUTEUR_GRILLE - 4
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

                if 0 <= nx < LARGEUR_GRILLE and 0 <= ny < HAUTEUR_GRILLE:
                    case = cases[ny][nx]
                    vision[0, iy, ix] = 1.0 if case["type"] == "pomme" else 0.0
                    vision[1, iy, ix] = get_light(temp, case)

        return torch.tensor(vision, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, 2, taille, taille]

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

        if "haut" in actions: dy -= 1
        if "bas" in actions: dy += 1
        if "gauche" in actions: dx -= 1
        if "droite" in actions: dx += 1

        oldx, oldy = self.x, self.y
        self.x = max(0, min(LARGEUR_GRILLE - 1, self.x + dx))
        self.y = max(0, min(HAUTEUR_GRILLE - 1, self.y + dy))

        # Manger
        if "manger" in actions:
            case = grille["cases"][self.y][self.x]
            if case["type"] == "pomme":
                case["type"] = None
                self.faim = max(0, self.faim - 0.4)
                self.nb_pommes_mangees += 1
                self.nb_pommes -= 1

        # Dégâts selon lumière
        lumi = get_light(grille["temps"], grille["cases"][self.y][self.x])
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

    def mutate(self, taux):
        for param in self.parameters():
            with torch.no_grad():
                mutation_mask = torch.rand_like(param) < taux
                param += mutation_mask * torch.randn_like(param) * taux

    def forward(self, vision_tensor):
        x = self.conv(vision_tensor)
        x = x.view(x.size(0), -1)
        return self.fc(x)






def simuler_multi(fouloide, seeds, n_ticks=300):
    total_score = 0

    for seed in seeds:
        fouloide = deepcopy(fouloide)  # ← Cloner à chaque test pour éviter toute mémoire ou état résiduel
        py_rng = random.Random(seed)
        rng = np.random.RandomState(seed)
        grille = generer_grille(rng, py_rng)
        fouloide.reset()
        historique_lumieres = []

        while not fouloide.est_mort and fouloide.ticks < n_ticks:
            fouloide.nb_pommes = ajouter_pomme(grille, fouloide.nb_pommes, 30, rng, py_rng)
            mise_a_jour_luminosite(grille)
            fouloide.agir(grille)

            # Mesure de la lumière locale
            lumi = get_light(grille["temps"], grille["cases"][fouloide.y][fouloide.x])
            historique_lumieres.append(lumi)

        total_score += (
            fouloide.nb_pommes_mangees * 10
            + fouloide.ticks
            + sum(1 if l >= 0.5 else (0.5 if l >= 0.3 else -1) for l in historique_lumieres)
            + (50 if not fouloide.est_mort else 0)
        )

    return total_score / len(seeds)


def charger_meilleur(path="meilleur.pt"):
    try:
        model = NeuralFouloide()
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        return model
    except FileNotFoundError:
        print("[!] Aucun fichier meilleur.pt trouvé.")
        return None






def crossover(parent1: NeuralFouloide, parent2: NeuralFouloide) -> NeuralFouloide:
    enfant = copy.deepcopy(parent1)  # On part du modèle du parent1

    for p_enf, p1, p2 in zip(enfant.parameters(), parent1.parameters(), parent2.parameters()):
        mask = torch.rand_like(p_enf) < 0.5  # Masque aléatoire 0/1
        with torch.no_grad():
            p_enf.copy_(torch.where(mask, p1.data, p2.data))  # choix gène à gène

    return enfant



SEEDS = [random.randint(0, 10000) for _ in range(3)]
compteur_seed = 0

if __name__ == "__main__":

    meilleur = charger_meilleur()
    population = []






    # Compléter la population avec des individus aléatoires
    while len(population) < TAILLE_POPULATION:
        if meilleur:
            population.append(deepcopy(meilleur))


        else:
            population.append(NeuralFouloide())

    taux_mutation = TAUX_MUTATION
    historique_scores = []
    hall_of_fame = []  # [(score, individu)]
    HALL_OF_FAME_MAX = 5

    

    for generation in range(NB_GENERATIONS):
        
        start_gen = time.perf_counter()
        print(f"Génération {generation}")

        # Parallélisation des simulations
        if compteur_seed % 10 == 0:
            SEEDS = [random.randint(0, 10000) for _ in range(3)]
        compteur_seed += 1

        with mp.Pool() as pool:

            chunksize = TAILLE_POPULATION // (mp.cpu_count() * 2)
            population_valide = []
            for ind in population:
                population_valide.append(ind)

            if len(population_valide) < TAILLE_POPULATION:
                print(f"[!] {TAILLE_POPULATION - len(population_valide)} individus invalides supprimés")
                while len(population_valide) < TAILLE_POPULATION:
                    population_valide.append(NeuralFouloide())

            scores = [simuler_multi(ind, SEEDS, NB_TICKS) for ind in population_valide]




            scores_population = list(zip(scores, population))
            scores_population.sort(reverse=True, key=lambda x: x[0])

            scores_gen = [s for s, _ in scores_population]
            historique_best.append(max(scores_gen))
            historique_moyenne.append(np.mean(scores_gen))

            best_score, best = scores_population[0]
            reeval = simuler_multi(deepcopy(best), SEEDS, NB_TICKS)
            best_score = reeval  # ← on remplace l'ancien score incertain
            best = deepcopy(best)  # ← pour éviter toute modification ultérieure

            # --- Sélection par tournoi ---
            def tournoi_selection(population, scores, taille_tournoi=5, nb_gagnants=10):
                gagnants = []
                for _ in range(nb_gagnants):
                    participants = random.sample(list(zip(scores, population)), taille_tournoi)
                    gagnant = max(participants, key=lambda x: x[0])[1]
                    gagnants.append(gagnant)
                return gagnants

            top_individus = tournoi_selection(population, scores, taille_tournoi=5, nb_gagnants=10)


            # Suivi du score et Hall of Fame
            historique_scores.append(best_score)

            # Ajout au Hall of Fame si meilleur que les pires du HoF
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

            # Ajustement mutation
            # Paramètres adaptatifs
            MUTATION_BASE = 0.2
            MUTATION_MIN = 0.05
            MUTATION_MAX = 0.6
            STAGNATION_SEUIL = 0.01  # variation < 1%
            NB_STAGNATION_MAX = 5
            variation = (best_score - moyenne_prec) / moyenne_prec if moyenne_prec > 0 else 0

            # Mémorisation du nombre de générations stagnantes
            if 'nb_stagnantes' not in locals():
                nb_stagnantes = 0

            if abs(variation) < STAGNATION_SEUIL:
                nb_stagnantes += 1
            else:
                nb_stagnantes = 0

            # Logique d'ajustement
            # Ajustement adaptatif du taux de mutation
            if nb_stagnantes >= NB_STAGNATION_MAX:
                # En cas de stagnation prolongée : on augmente franchement le taux
                taux_mutation = min(taux_mutation + 0.05, MUTATION_MAX)
            elif variation > 0.05:
                # Bonne amélioration : on réduit légèrement le taux
                taux_mutation = max(taux_mutation - 0.02, MUTATION_MIN)
            else:
                # Sinon, retour progressif vers MUTATION_BASE
                taux_mutation += (MUTATION_BASE - taux_mutation) * 0.1

            # Clamp final
            taux_mutation = np.clip(taux_mutation, MUTATION_MIN, MUTATION_MAX)

            # Nouvelle génération avec élitisme
            # Injection des meilleurs du Hall of Fame
            hof_individus = [x[1] for x in hall_of_fame]
            next_population = [deepcopy(x[1]) for x in hall_of_fame]
            next_population.append(deepcopy(best))  # forcer le best actuel


            while len(next_population) < TAILLE_POPULATION:
                p1, p2 = random.sample(top_individus, 2)
                enfant = crossover(p1, p2)
                enfant.mutate(taux_mutation)
                next_population.append(enfant)

            population = next_population

            duration = time.perf_counter() - start_gen

            # Sauvegarde
            best_clone = deepcopy(best)
            torch.save(enfant.state_dict(), "meilleur.pt")



            nb_params = sum(p.numel() for p in best.parameters())

            print(f">>> Sauvegarde du meilleur avec score {best_score} | Taux mutation : {taux_mutation:.4f} terminée en {duration:.2f} secondes | neuronnes : {nb_params}")
            #with open("log.txt", "a") as f:
            #    f.write(f"Génération {generation + 1} - Meilleur score : {best_score:.4f} - Taux mutation : {taux_mutation:.4f}\n terminée en {duration:.2f} secondes")
            #    for idx, (score, _) in enumerate(scores_population):
            #        f.write(f"  Individu {idx + 1:03} - Score : {score:.4f}\n")




            if generation % 10 == 0 or generation == NB_GENERATIONS - 1:
                plt.clf()
                plt.plot(historique_best, label="Meilleur score", color="green")
                plt.plot(historique_moyenne, label="Score moyen", color="blue")
                plt.xlabel("Génération")
                plt.ylabel("Score")
                plt.title("Évolution des scores")
                plt.legend()
                plt.pause(0.01)

