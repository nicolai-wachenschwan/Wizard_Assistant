import random
import copy
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import math

###########
# Hilfsklassen für das Wizard-Spiel
###########
from wizard_core_game_classes import *

###########
# Bayes-Schätzung der Rewards
###########
from simulator import PureMonteCarloSimulator

class BayesianRewardEstimator:
    """
    Schätzt Rewards pro (Kontext, Karte) anhand eines Beta-Binomial-Modells.
    Wir normieren die Punkte auf [−200, 220] → [0, 1] für Beta, 
    und rekonstruieren sie später per Denormalisierung.
    """
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.context_observations: Dict[str, List[float]] = defaultdict(list)
        self.alpha_params: Dict[str, float] = defaultdict(lambda: self.prior_alpha)
        self.beta_params: Dict[str, float] = defaultdict(lambda: self.prior_beta)
        self.cached_estimates: Dict[Tuple[str,int], float] = {}
        self.cache_version: Dict[str, int] = defaultdict(int)
    
    def update_observation(self, context_key: str, reward: float):
        """
        Führt einen Beobachtungswert (Reward) in den Posterior ein.
        reward ∈ ℝ (z.B. −50 … +100), wir normieren intern auf [0,1] und updaten α/β.
        """
        normalized_reward = self._normalize_reward(reward)
        self.context_observations[context_key].append(reward)
        if normalized_reward > 0.5:
            self.alpha_params[context_key] += 1.0
        else:
            self.beta_params[context_key] += 1.0
        self.cache_version[context_key] += 1
        # Caches löschen, falls sich Posterior geändert hat
        cache_key = (context_key, self.cache_version[context_key])
        if cache_key in self.cached_estimates:
            del self.cached_estimates[cache_key]
    
    def _normalize_reward(self, reward: float) -> float:
        """
        Normiert Reward ∈ [min_r, max_r] auf [0,1].
        Wir nehmen an: min_r = −200, max_r = +220 (aus Erfahrungswerten).
        """
        min_r, max_r = -200.0, 220.0
        if max_r == min_r:
            return 0.5
        return max(0.0, min(1.0, (reward - min_r) / (max_r - min_r)))
    
    def _denormalize_estimate(self, normalized_estimate: float) -> float:
        """
        Mappt einen Normalwert ∈ [0,1] zurück auf [min_r, max_r].
        """
        min_r, max_r = -200.0, 220.0
        return min_r + normalized_estimate * (max_r - min_r)
    
    def get_posterior_parameters(self, context_key: str) -> Tuple[float, float]:
        return (self.alpha_params[context_key], self.beta_params[context_key])
    
    def estimate_reward(self, context_key: str) -> float:
        """
        Gibt den posterioren Erwartungswert (den denormalisierten Mittelwert) 
        anhand von α/(α+β) zurück.
        """
        cache_key = (context_key, self.cache_version[context_key])
        if cache_key in self.cached_estimates:
            return self.cached_estimates[cache_key]
        
        alpha, beta = self.get_posterior_parameters(context_key)
        if (alpha + beta) > 0:
            normalized_estimate = alpha / (alpha + beta)
        else:
            normalized_estimate = 0.5
        estimate = self._denormalize_estimate(normalized_estimate)
        self.cached_estimates[cache_key] = estimate
        return estimate
    
    def get_uncertainty(self, context_key: str) -> float:
        """
        Gibt die (approx.) Standardabweichung der posterior-Beta-Verteilung, 
        skaliert auf dieselbe Reward-Skala [−200, 220], zurück.
        """
        alpha, beta = self.get_posterior_parameters(context_key)
        if (alpha + beta) == 0 or (alpha + beta + 1) == 0:
            # Extremfall: keine Parameter → maximale Unsicherheit
            return self._denormalize_estimate(1.0) - self._denormalize_estimate(0.0)
        
        variance_norm = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        min_r, max_r = -200.0, 220.0
        stdev_scaled = math.sqrt(variance_norm) * (max_r - min_r)
        return stdev_scaled
    
    def get_confidence_interval(self, context_key: str, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Berechnet das (approx.) Beta‐Konfidenzintervall für den Posterior in [−200,220].
        """
        from scipy import stats  # erst importieren, wenn benötigt
        alpha, beta = self.get_posterior_parameters(context_key)
        if alpha + beta == 0:
            alpha = self.prior_alpha
            beta = self.prior_beta
        
        lower_norm = stats.beta.ppf((1 - confidence) / 2, alpha, beta)
        upper_norm = stats.beta.ppf((1 + confidence) / 2, alpha, beta)
        return (self._denormalize_estimate(lower_norm), self._denormalize_estimate(upper_norm))

###########
# Backpropagation mit zusätzlichem Decision-Quality-Faktor
###########

class BayesOptimalBackpropagator:
    """
    Propagiert den finalen Reward jedes Spielers über die Entscheidungspfade zurück
    und gewichtet jede Entscheidung zusätzlich nach "Qualität" (wie nahe am geschätzten Bestscore).
    """
    def __init__(self, reward_estimator: BayesianRewardEstimator,
                 temporal_discount: float = 0.95,
                 causal_weight_other_player: float = 0.8,
                 base_gamma: float = 1.0):
        self.reward_estimator = reward_estimator
        self.temporal_discount = temporal_discount
        self.causal_weight_other_player = causal_weight_other_player
        self.base_gamma = base_gamma  # Basis‐Gamma für decision_quality
    
    def propagate_decision_path(
        self,
        decision_path: List[Tuple[int, Card, str, List[Card]]],
        final_reward_for_target: float,
        target_player: int
    ):
        """
        decision_path: Liste von (spieler_id, karte, context_key, valid_actions).
        final_reward_for_target: Score des target_player am Rundeende.
        target_player: ID des Spielers, aus dessen Perspektive wir belohnen.
        """
        if not decision_path:
            return
        
        # 1) Temporal Discounts (für jede Position i im Pfad)
        discounted_rewards = self._calculate_temporal_discounts(decision_path, final_reward_for_target)
        # 2) Causal Weights: 1.0, wenn Entscheidung von target_player; sonst causal_weight_other_player
        causal_weights = self._calculate_causal_weights(decision_path, target_player)
        
        for i in range(len(decision_path)):
            player_i, card_i, context_key_i, valid_actions_i = decision_path[i]
            discounted_reward = discounted_rewards[i]
            causal_w = causal_weights[i]
            
            # --- ENTSCHEIDUNGSQUALITÄT berechnen ---
            # a) Schätze Reward für jede gültige Karte in diesem Kontext
            scores: Dict[Card, float] = {}
            for c in valid_actions_i:
                card_ctx_key = f"{context_key_i}_card_{c}"
                scores[c] = self.reward_estimator.estimate_reward(card_ctx_key)
            
            # b) Finde "beste" Karte laut aktueller Schätzung
            best_card, best_score = max(scores.items(), key=lambda t: t[1])
            chosen_score = scores[card_i]
            delta = best_score - chosen_score  # Wie viel schlechter war die gewählte Karte
            
            # c) Bestimme eine skalierte Anzahl Beobachtungen für Wahl und beste Karte
            ctx_chosen = f"{context_key_i}_card_{card_i}"
            ctx_best = f"{context_key_i}_card_{best_card}"
            obs_chosen = len(self.reward_estimator.context_observations.get(ctx_chosen, []))
            obs_best = len(self.reward_estimator.context_observations.get(ctx_best, []))
            obs_count = obs_chosen + obs_best  # Gesamtbeobachtungen, um Gamma zu skalieren
            
            # d) Skaliere Gamma in Abhängigkeit von obs_count:
            #    scale = obs_count / (obs_count + k) mit k=5 → anfänglich scale≈0, später scale→1
            k = 5.0
            scale = obs_count / (obs_count + k) if (obs_count + k) > 0 else 0.0
            gamma = self.base_gamma * scale
            
            # e) Bestimme Unsicherheit (stddev) im Kontext der gewählten Karte:
            uncertainty = self.reward_estimator.get_uncertainty(ctx_chosen)
            
            # f) Wenn Delta unter Unsicherheit, kein Abstrafen; sonst exponential
            if delta < uncertainty:
                decision_quality_weight = 1.0
            else:
                # Wir ziehen Uncertainty ab, damit nur *überschüssiges* Delta bestraft wird
                decision_quality_weight = math.exp(-gamma * (delta - uncertainty))
            
            # --- Gesamtgewicht und Update ---
            final_weighted_reward = discounted_reward * causal_w * decision_quality_weight
            
            # 1) Aktualisiere Reward‐Estimator für (context_key_i, card_i)
            context_card_key = f"{context_key_i}_card_{card_i}"
            self.reward_estimator.update_observation(context_card_key, final_weighted_reward)
            # 2) Transfer-Learning auf reinen Kontext (halbes Gewicht)
            self.reward_estimator.update_observation(context_key_i, final_weighted_reward * 0.5)
    
    def _calculate_temporal_discounts(
        self,
        decision_path: List[Tuple[int, Card, str, List[Card]]],
        final_reward: float
    ) -> List[float]:
        """
        Berechnet für jeden Schritt i: final_reward × (temporal_discount)^(#Schritte von Ende).
        """
        discounted: List[float] = []
        path_length = len(decision_path)
        for i in range(path_length):
            steps_from_end = (path_length - 1 - i)
            discount_factor = self.temporal_discount ** steps_from_end
            discounted.append(final_reward * discount_factor)
        return discounted
    
    def _calculate_causal_weights(
        self,
        decision_path: List[Tuple[int, Card, str, List[Card]]],
        target_player: int
    ) -> List[float]:
        """
        Falls die Entscheidung von target_player stammt → Gewicht 1.0,
        sonst → causal_weight_other_player (z.B. 0.8).
        """
        weights: List[float] = []
        for player_id, _, _, _ in decision_path:
            if player_id == target_player:
                weights.append(1.0)
            else:
                weights.append(self.causal_weight_other_player)
        return weights

###########
# CardEvaluator
###########

class BayesOptimalCardEvaluator:
    """
    Bewertet zu einem aktuellen Spielzustand: Für jeden möglichen Spielzug (jede Karte),
    schätzt er Reward + Unsicherheit und berechnet einen UCB-Score:
      Score = estimated_reward + exploration_factor * uncertainty
    """
    def __init__(self, reward_estimator: BayesianRewardEstimator,
                 exploration_factor: float = 0.5):
        self.reward_estimator = reward_estimator
        self.exploration_factor = exploration_factor
    
    def evaluate_card(
        self,
        player: int,
        card: Card,
        game_state: GameState
    ) -> float:
        context_key = self._create_evaluation_context(player, game_state, game_state.current_trick_cards)
        card_context_key = f"{context_key}_card_{card}"
        est = self.reward_estimator.estimate_reward(card_context_key)
        unc = self.reward_estimator.get_uncertainty(card_context_key)
        exploration_bonus = self.exploration_factor * unc
        return est + exploration_bonus
    
    def _create_evaluation_context(
        self,
        player: int,
        game_state: GameState,
        trick_cards_in_progress: List[Tuple[int, Card]]
    ) -> str:
        bid_diff = game_state.bids[player] - game_state.tricks_won.get(player, 0)
        remaining_tricks = game_state.round_number - game_state.current_trick
        trick_position = len(trick_cards_in_progress)
        has_trump = 1 if game_state.trump_suit is not None else 0
        return f"p{player}_bd{bid_diff}_rt{remaining_tricks}_tp{trick_position}_ht{has_trump}"

###########
# CardEvaluator mit Ähnlichkeitsschätzung
###########

class ExpectedRewardCardEvaluator:
    """
    Bewertet zu einem aktuellen Spielzustand jede mögliche Karte.
    Die Bewertung basiert auf dem geschätzten Erwartungswert (Reward).
    Um die Schätzung für seltene/unbekannte Karten zu verbessern, wird eine
    Ähnlichkeitsschätzung implementiert: Der Reward wird als gewichteter
    Mittelwert aus dem spezifischen Reward der Karte und dem allgemeinen
    Reward des Kontexts berechnet.
    """
    def __init__(self, reward_estimator: BayesianRewardEstimator, skepticism: int = 5):
        self.reward_estimator = reward_estimator
        # Der skepticism-Faktor 'k' bestimmt, nach wie vielen Beobachtungen
        # eine spezifische Schätzung zu 50% gewichtet wird.
        self.k = skepticism

    def evaluate_card(
        self,
        player: int,
        card: Card,
        game_state: GameState
    ) -> float:
        """
        Berechnet den erwarteten Reward für das Spielen einer Karte durch
        eine gewichtete Mischung aus spezifischer und allgemeiner Kontextschätzung.
        """
        # 1. Schlüssel für den allgemeinen und spezifischen Kontext erstellen
        general_context_key = self._create_evaluation_context(player, game_state, game_state.current_trick_cards)
        card_context_key = f"{general_context_key}_card_{card}"

        # 2. Rohe Reward-Schätzungen für beide Kontexte abrufen
        specific_reward = self.reward_estimator.estimate_reward(card_context_key)
        general_reward = self.reward_estimator.estimate_reward(general_context_key)

        # 3. Anzahl der Beobachtungen für den spezifischen Kontext ermitteln
        num_observations = len(self.reward_estimator.context_observations.get(card_context_key, []))

        # 4. Gewicht für die spezifische Schätzung berechnen
        # Die Formel N / (N + k) nähert sich 1 an, je mehr Beobachtungen (N) es gibt.
        # Bei N=0 ist das Gewicht 0. Bei N=k ist es 0.5.
        specific_weight = num_observations / (num_observations + self.k)

        # 5. Endgültigen Reward als gewichteten Mittelwert berechnen
        # Wenn die Karte unbekannt ist (N=0), wird zu 100% der allgemeine Reward genutzt.
        # Wenn die Karte sehr gut bekannt ist (N groß), wird fast nur ihr spezifischer Reward genutzt.
        final_estimated_reward = (specific_weight * specific_reward) + ((1 - specific_weight) * general_reward)
        
        return final_estimated_reward

    def _create_evaluation_context(
        self,
        player: int,
        game_state: GameState,
        trick_cards_in_progress: List[Tuple[int, Card]]
    ) -> str:
        # Diese Hilfsfunktion bleibt unverändert
        bid_diff = game_state.bids.get(player, 0) - game_state.tricks_won.get(player, 0)
        remaining_tricks = game_state.round_number - game_state.current_trick
        trick_position = len(trick_cards_in_progress)
        has_trump = 1 if game_state.trump_suit is not None else 0
        return f"p{player}_bd{bid_diff}_rt{remaining_tricks}_tp{trick_position}_ht{has_trump}"
###########
# Hauptklasse: Bayes‐optimale Empfehlung
###########

class BayesOptimalWizardAssistant:
    """
    Dieser Assistent führt für eine gegebene Spielsituation (GameState) 
    eine Anzahl Monte Carlo Simulationen durch, füttert den BayesLearner
    mit den Entscheidungspfaden und deren Rewards und gibt dann
    für jeden möglichen Spielzug eine UCB‐bewertete Empfehlung zurück.
    """
    def __init__(self, num_simulations: int = 1000, num_distributions: int = 100):
        self.num_simulations = num_simulations
        self.num_distributions = num_distributions
        self.simulations_per_distribution = max(1, num_simulations // num_distributions)
        
        # Bayes-Learner (prior = 2.0,2.0, damit nicht von vornherein zu extrem)
        self.reward_estimator = BayesianRewardEstimator(prior_alpha=2.0, prior_beta=2.0)
        self.backpropagator = BayesOptimalBackpropagator(self.reward_estimator,
                                                         temporal_discount=0.95,
                                                         causal_weight_other_player=0.8,
                                                         base_gamma=1.0)
        self.card_evaluator = ExpectedRewardCardEvaluator(self.reward_estimator, skepticism=5)
    
    def get_recommendation(
        self,
        game_state: GameState,
        player_id: int
    ) -> Dict[Card, float]:
        """
        Generiere für `player_id` in der aktuellen Situation `game_state`
        eine Empfehlung: Karte → UCB‐Score.
        """
        available_cards = self.get_available_cards(game_state, player_id)
        if not available_cards:
            return {}
        
        # Alle unbekannten Karten ermitteln (die nicht im eigenen Blatt, nicht auf Tisch, nicht gespielt)
        unknown_cards = self._get_unknown_cards(game_state, player_id)
        
        # Wir sammeln Rewards pro Karte (rein fürs Debugging bzw. Statistik, nicht direkt für finalen Score)
        card_rewards_for_player: Dict[Card, List[int]] = {c: [] for c in available_cards}
        
        # 1) Loop über verschiedene Zufalls-„Verteilungen“ der unbekannten Karten
        for dist_idx in range(self.num_distributions):
            if dist_idx > 0 and dist_idx % (self.num_distributions // 10 if self.num_distributions >= 10 else 1) == 0:
                print(f"Progress: {dist_idx}/{self.num_distributions} distributions")
            curr_unknown = unknown_cards.copy()
            random.shuffle(curr_unknown)
            
            # 2) Simuliere für diesen Karten‐Shuffle _simulate_one_card_distribution
            self._simulate_one_card_distribution(
                game_state,
                player_id,
                curr_unknown,
                available_cards,
                card_rewards_for_player
            )
        
        # 3) Am Ende wertet der CardEvaluator (UCB‐Formel) aus, ohne die direkten Rewards zu mitteln.
        final_scores: Dict[Card, float] = {}
        for c in available_cards:
            final_scores[c] = self.card_evaluator.evaluate_card(player_id, c, game_state)
        return final_scores
    
    def _simulate_one_card_distribution(
        self,
        original_game_state: GameState,
        player_id: int,
        distributed_unknown_cards: List[Card],
        available_cards: List[Card],
        card_rewards_accumulator: Dict[Card, List[int]]
    ):
        """
        Für eine festgelegte Permutation der unbekannten Karten (distributed_unknown_cards):
        - Spiele n = simulations_per_distribution Runden, in denen `player_id` in der
          Ausgangssituation je eine der `available_cards` testet.
        - Sammle den finalen Reward für player_id (für Statistik) und führe das
          Backpropagations‐Update für *alle* Spieler durch.
        """
        for card_to_test in available_cards:
            # a) Kontext, in dem player_id diese Karte jetzt spielt:
            context_of_choice = self.card_evaluator._create_evaluation_context(
                player_id, original_game_state, original_game_state.current_trick_cards
            )
            # b) Erzeuge neuen Zustand *nach* dem Spielen von card_to_test
            state_after = self._create_state_with_played_card(
                original_game_state, player_id, card_to_test
            )

            # FIX: For mid-round simulations, ensure bids are populated if they are missing.
            # This handles test cases or scenarios where get_recommendation is called
            # on a state without complete bid information.
            if len(state_after.bids) < len(state_after.players):
                for p in state_after.players:
                    if p not in state_after.bids:
                        state_after.bids[p] = random.randint(0, state_after.round_number)
            
            # c) Nun simulations_per_distribution-mal den Rest der Runde zufällig spielen
            for _ in range(self.simulations_per_distribution):
                # Reinen Monte Carlo Simulator instanziieren
                simulator = PureMonteCarloSimulator(state_after)
                sim_bids, sim_tricks_won, decision_path = simulator.simulate_random_game(distributed_unknown_cards.copy())

                # Prepend: unsere getestete Entscheidung zu Beginn
                full_path: List[Tuple[int, Card, str, List[Card]]] = [
                    (player_id, card_to_test, context_of_choice,
                     # Welche valid_actions gab es zum Zeitpunkt, als player_id diese Karte wählte?
                     # Wir müssen sie aus dem ursprünglichen Zustand neu berechnen:
                     self._get_valid_actions_for_context(player_id, original_game_state)
                    )
                ] + decision_path

                # d) Reward des Testspielers in dieser Simulation
                reward_test = WizardRules.calculate_score(
                    sim_bids[player_id], sim_tricks_won.get(player_id, 0)
                )
                card_rewards_accumulator[card_to_test].append(reward_test)

                # e) Backpropagation: für jeden Spieler p_in_game
                for p_in_game in state_after.players:
                    reward_p = WizardRules.calculate_score(
                        sim_bids[p_in_game], sim_tricks_won.get(p_in_game, 0)
                    )
                    self.backpropagator.propagate_decision_path(full_path, reward_p, p_in_game)

    def get_available_cards(
        self,
        game_state: GameState,
        player_id: int
    ) -> List[Card]:
        """
        Öffentliche Methode: Gibt alle validen Karten aus der Hand von player_id 
        im momentanen Zustand zurück.
        """
        return self._get_valid_cards(game_state, player_id)
    
    def _get_valid_cards(
        self,
        game_state: GameState,
        player_id: int
    ) -> List[Card]:
        """
        Interne Helfer­Methode: Listet alle Karten in player_id.hands,
        die laut WizardRules im aktuellen Stich gültig sind.
        """
        hand = game_state.hands.get(player_id, [])
        if not hand:
            return []
        return [
            c for c in hand
            if WizardRules.is_valid_play(c, hand, game_state.current_trick_cards, game_state.trump_suit)
        ]
    
    def _get_unknown_cards(self, game_state: GameState, player_id: int) -> List[Card]:
        """
        Ermittelt alle Karten, die Player(player_id) nicht kennt:
        - Nicht in player_id.hands
        - Nicht in game_state.played_cards
        - Nicht auf dem Tisch (current_trick_cards)
        """
        all_deck = set(WizardDeck().cards)
        known = game_state.played_cards.copy()
        for _, c in game_state.current_trick_cards:
            known.add(c)
        own_hand = set(game_state.hands.get(player_id, []))
        known_to_player = known.union(own_hand)
        return list(all_deck - known_to_player)
    
    def _create_state_with_played_card(
        self, game_state: GameState, player_id: int, card_played: Card
    ) -> GameState:
        """
        Erzeugt eine Kopie von game_state, in der player_id sofort `card_played`
        in den aktuellen Stich ablegt. current_player wird auf den Nachfolger gesetzt.
        (current_trick selbst bleibt gleich; der Stich endet erst, wenn alle Spieler gespielt haben.)
        """
        new_state = game_state.copy()
        if card_played in new_state.hands[player_id]:
            new_state.hands[player_id].remove(card_played)
        new_state.current_trick_cards.append((player_id, card_played))
        new_state.played_cards.add(card_played)
        idx = new_state.players.index(player_id)
        new_state.current_player = new_state.players[(idx + 1) % len(new_state.players)]
        return new_state
    
    def _get_valid_actions_for_context(
        self, player_id: int, game_state: GameState
    ) -> List[Card]:
        """
        Ermittelt (rein deterministisch) die validen Karten, 
        die player_id im gegebenen game_state hätte spielen können.
        Wird benötigt, um in full_path das 4-te Feld `valid_actions` zu füllen,
        falls wir dort nicht schon den originalen List‐Eintrag aus Simulator haben.
        """
        hand = game_state.hands.get(player_id, [])
        if not hand:
            return []
        return [
            c for c in hand
            if WizardRules.is_valid_play(c, hand, game_state.current_trick_cards, game_state.trump_suit)
        ]
    
    def get_detailed_analysis(
        self, game_state: GameState, player_id: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Liefert für jede valide Karte als String:
        - geschätzter Reward
        - Unsicherheit
        - Konfidenzintervall
        - Anzahl Beobachtungen
        - finalen UCB-Score (Recommendation)
        """
        available = self.get_available_cards(game_state, player_id)
        analysis: Dict[str, Dict[str, float]] = {
            'card_evaluations': {},
            'confidence_intervals': {},
            'uncertainties': {},
            'observation_counts': {},
            'recommendations': {}
        }
        for c in available:
            context_key = self.card_evaluator._create_evaluation_context(
                player_id, game_state, game_state.current_trick_cards
            )
            card_ctx_key = f"{context_key}_card_{c}"
            est = self.reward_estimator.estimate_reward(card_ctx_key)
            unc = self.reward_estimator.get_uncertainty(card_ctx_key)
            ci = self.reward_estimator.get_confidence_interval(card_ctx_key)
            obs = len(self.reward_estimator.context_observations.get(card_ctx_key, []))
            analysis['card_evaluations'][str(c)] = est
            analysis['uncertainties'][str(c)] = unc
            analysis['confidence_intervals'][str(c)] = ci
            analysis['observation_counts'][str(c)] = obs
            rec_score = self.card_evaluator.evaluate_card(player_id, c, game_state)
            analysis['recommendations'][str(c)] = rec_score
        return analysis




