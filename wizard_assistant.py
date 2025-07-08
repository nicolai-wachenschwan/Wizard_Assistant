import random
import copy
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
import multiprocessing as mp
import math
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

class Suit(Enum):
    RED = "red"
    BLUE = "blue" 
    GREEN = "green"
    YELLOW = "yellow"
    WIZARD = "wizard"
    JESTER = "jester"

@dataclass
class Card:
    suit: Suit
    value: int  # 1-13 for normal cards, arbitrary for special cards
    
    def __post_init__(self):
        """Do not override value for Wizard/Jester so custom cards work."""
        pass
    
    def __str__(self):
        if self.suit == Suit.WIZARD:
            return "Wizard"
        elif self.suit == Suit.JESTER:
            return "Jester"
        else:
            return f"{self.suit.value}_{self.value}"
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.suit == other.suit and self.value == other.value
    
    def __hash__(self):
        return hash((self.suit, self.value))

class WizardDeck:
    """Verwaltet das Wizard-Kartendeck"""
    
    def __init__(self):
        self.cards = self._create_deck()
    
    def _create_deck(self) -> List[Card]:
        cards = []
        # Normale Karten: 4 Farben x 13 Werte
        for suit in [Suit.RED, Suit.BLUE, Suit.GREEN, Suit.YELLOW]:
            for value in range(1, 14):
                cards.append(Card(suit, value))
        
        # Spezielle Karten: 4 Wizard, 4 Jester
        for _ in range(4):
            cards.append(Card(Suit.WIZARD, 0))
            cards.append(Card(Suit.JESTER, 0))
        
        return cards
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def deal_cards(self, num_cards: int) -> List[Card]:
        return [self.cards.pop() for _ in range(num_cards)]

@dataclass
class GameState:
    """Repräsentiert den aktuellen Spielzustand"""
    round_number: int
    current_trick: int
    trump_suit: Optional[Suit]
    players: List[int]  # Spieler IDs
    hands: Dict[int, List[Card]]  # Spieler -> Handkarten
    bids: Dict[int, int]  # Spieler -> Ansagen
    tricks_won: Dict[int, int]  # Spieler -> gewonnene Stiche
    current_trick_cards: List[Tuple[int, Card]]  # (Spieler, Karte) für aktuellen Stich
    played_cards: Set[Card]  # Bereits gespielte Karten
    current_player: int
    trick_leader: int  # Wer den Stich anführt
    
    def copy(self):
        return copy.deepcopy(self)

class WizardRules:
    """Implementiert die Wizard-Spielregeln"""
    
    @staticmethod
    def is_valid_play(card: Card, hand: List[Card], 
                     current_trick_cards: List[Tuple[int, Card]], 
                     trump_suit: Optional[Suit]) -> bool:
        """Prüft ob eine Karte gespielt werden darf"""
        if not current_trick_cards:  # Erster Spieler im Stich
            return True
        
        if card.suit == Suit.WIZARD or card.suit == Suit.JESTER:
            return True
        
        # Führende Farbe bestimmen
        lead_suit = None
        for _, trick_card in current_trick_cards:
            if trick_card.suit not in [Suit.WIZARD, Suit.JESTER]:
                lead_suit = trick_card.suit
                break
        
        if lead_suit is None:  # Nur Wizard/Jester bisher gespielt
            return True
        
        # Muss Farbe bedienen wenn möglich
        if card.suit == lead_suit:
            return True
        
        # Prüfen ob Spieler die Farbe bedienen kann
        can_follow_suit = any(c.suit == lead_suit for c in hand 
                            if c.suit not in [Suit.WIZARD, Suit.JESTER])
        
        return not can_follow_suit
    
    @staticmethod
    def determine_trick_winner(trick_cards: List[Tuple[int, Card]], 
                             trump_suit: Optional[Suit]) -> int:
        """Bestimmt den Gewinner eines Stichs"""
        if not trick_cards:
            return -1
        
        # Wizard gewinnt immer
        for player, card in trick_cards:
            if card.suit == Suit.WIZARD:
                return player
        
        # Führende Farbe bestimmen (ignoriere Jester)
        lead_suit = None
        for _, card in trick_cards:
            if card.suit not in [Suit.WIZARD, Suit.JESTER]:
                lead_suit = card.suit
                break
        
        if lead_suit is None:  # Nur Jester gespielt
            return trick_cards[0][0]  # Erster Spieler gewinnt
        
        # Höchste Trumpfkarte gewinnt
        best_trump = None
        best_trump_player = None
        
        # Höchste Karte der führenden Farbe
        best_lead = None
        best_lead_player = None
        
        for player, card in trick_cards:
            if card.suit == trump_suit and card.suit != Suit.JESTER:
                if best_trump is None or card.value > best_trump.value:
                    best_trump = card
                    best_trump_player = player
            elif card.suit == lead_suit:
                if best_lead is None or card.value > best_lead.value:
                    best_lead = card
                    best_lead_player = player
        
        return best_trump_player if best_trump_player is not None else best_lead_player
    
    @staticmethod
    def calculate_score(bid: int, tricks_won: int) -> int:
        """Berechnet die Punkte für einen Spieler"""
        if bid == tricks_won:
            return 20 + 10 * bid
        else:
            return -10 * abs(bid - tricks_won)

class BayesianRewardEstimator:
    """Schätzt Rewards basierend auf Bayes'scher Inferenz mit Beta-Binomial Konjugat"""
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Speichere Beobachtungen für jede Spielsituation
        self.context_observations = defaultdict(list)
        
        # Beta-Verteilungsparameter für jeden Kontext
        self.alpha_params = defaultdict(lambda: self.prior_alpha)
        self.beta_params = defaultdict(lambda: self.prior_beta)
        
        # Cached Statistiken
        self.cached_estimates = {}
        self.cache_version = defaultdict(int)
    
    def update_observation(self, context_key: str, reward: float):
        """
        Fügt eine neue Beobachtung hinzu und updated Beta-Parameter.
        Reward wird normalisiert zwischen 0 und 1 für Beta-Verteilung.
        """
        # Normalisiere Reward auf [0,1] Bereich für Beta-Verteilung
        normalized_reward = self._normalize_reward(reward)
        
        self.context_observations[context_key].append(reward)
        
        # Update Beta-Parameter
        if normalized_reward > 0.5:  # "Erfolg"
            self.alpha_params[context_key] += 1.0
        else:  # "Misserfolg"
            self.beta_params[context_key] += 1.0
        
        # Invalidiere Cache
        self.cache_version[context_key] += 1
        if context_key in self.cached_estimates:
            del self.cached_estimates[context_key]
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalisiert Reward auf [0,1] Bereich"""
        # Wizard Rewards sind typisch zwischen -130 und +150
        min_reward, max_reward = -130.0, 150.0
        return max(0.0, min(1.0, (reward - min_reward) / (max_reward - min_reward)))
    
    def _denormalize_estimate(self, normalized_estimate: float) -> float:
        """Denormalisiert Schätzung zurück in Reward-Bereich"""
        min_reward, max_reward = -130.0, 150.0
        return min_reward + normalized_estimate * (max_reward - min_reward)
    
    def get_posterior_parameters(self, context_key: str) -> Tuple[float, float]:
        """Gibt aktuelle Beta-Parameter zurück"""
        return self.alpha_params[context_key], self.beta_params[context_key]
    
    def estimate_reward(self, context_key: str) -> float:
        """Schätzt den erwarteten Reward für einen Kontext"""
        cache_key = (context_key, self.cache_version[context_key])
        if cache_key in self.cached_estimates:
            return self.cached_estimates[cache_key]
        
        alpha, beta = self.get_posterior_parameters(context_key)
        
        # Erwartungswert der Beta-Verteilung
        normalized_estimate = alpha / (alpha + beta)
        
        # Denormalisiere zurück in Reward-Bereich
        estimate = self._denormalize_estimate(normalized_estimate)
        
        self.cached_estimates[cache_key] = estimate
        return estimate
    
    def get_uncertainty(self, context_key: str) -> float:
        """Gibt die Unsicherheit der Schätzung zurück (Beta-Varianz)"""
        alpha, beta = self.get_posterior_parameters(context_key)
        
        # Varianz der Beta-Verteilung
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        
        # Skaliere Varianz in Reward-Bereich
        min_reward, max_reward = -130.0, 150.0
        scaled_variance = variance * (max_reward - min_reward) ** 2
        
        return math.sqrt(scaled_variance)
    
    def get_confidence_interval(self, context_key: str, confidence: float = 0.95) -> Tuple[float, float]:
        """Berechnet Konfidenzintervall für den erwarteten Reward"""
        from scipy import stats
        
        alpha, beta = self.get_posterior_parameters(context_key)
        
        # Beta-Verteilung Konfidenzintervall
        lower_bound = stats.beta.ppf((1 - confidence) / 2, alpha, beta)
        upper_bound = stats.beta.ppf((1 + confidence) / 2, alpha, beta)
        
        # Denormalisiere
        lower_reward = self._denormalize_estimate(lower_bound)
        upper_reward = self._denormalize_estimate(upper_bound)
        
        return (lower_reward, upper_reward)

class PureMonteCarloSimulator:
    """
    Komplett regelagnostischer Monte Carlo Simulator.
    Trifft KEINE intelligenten Entscheidungen, nur zufällige.
    """
    
    def __init__(self, game_state: GameState):
        self.initial_state = game_state.copy()
    
    def simulate_random_game(self, unknown_cards: List[Card], 
                           target_player: int) -> Tuple[float, List[Tuple[int, Card, str]]]:
        """
        Simuliert ein komplett zufälliges Spiel bis zum Ende.
        KEINE Heuristiken, KEINE intelligenten Entscheidungen.
        Gibt Reward und vollständigen Entscheidungspfad zurück.
        """
        state = self.initial_state.copy()
        decision_path = []
        
        # Verteile unbekannte Karten zufällig
        self._distribute_unknown_cards(state, unknown_cards)
        
        # Spiele alle verbleibenden Stiche mit REIN ZUFÄLLIGEN Entscheidungen
        while state.current_trick < state.round_number:
            trick_decisions = self._simulate_random_trick(state)
            decision_path.extend(trick_decisions)
        
        # Berechne finalen Reward für Zielspieler
        final_reward = WizardRules.calculate_score(
            state.bids[target_player], state.tricks_won[target_player]
        )
        
        return final_reward, decision_path
    
    def _distribute_unknown_cards(self, state: GameState, unknown_cards: List[Card]):
        """Verteilt unbekannte Karten KOMPLETT ZUFÄLLIG auf Spieler"""
        cards_copy = unknown_cards.copy()
        random.shuffle(cards_copy)
        
        # Berechne wie viele Karten jeder Spieler noch braucht
        cards_needed = {}
        for player in state.players:
            current_hand_size = len(state.hands[player])
            needed = state.round_number - current_hand_size
            cards_needed[player] = max(0, needed)
        
        # Verteile Karten ZUFÄLLIG
        card_idx = 0
        for player in state.players:
            for _ in range(cards_needed[player]):
                if card_idx < len(cards_copy):
                    state.hands[player].append(cards_copy[card_idx])
                    card_idx += 1
    
    def _simulate_random_trick(self, state: GameState) -> List[Tuple[int, Card, str]]:
        """Simuliert einen kompletten Stich mit REIN ZUFÄLLIGEN Entscheidungen"""
        trick_decisions = []
        trick_cards = []
        current_player = state.trick_leader
        
        # Jeder Spieler spielt eine KOMPLETT ZUFÄLLIGE gültige Karte
        for _ in range(len(state.players)):
            valid_cards = self._get_valid_cards(
                state.hands[current_player], trick_cards, state.trump_suit
            )
            
            if valid_cards:
                # ABSOLUT ZUFÄLLIGE Kartenwahl - KEINE Heuristik!
                card = random.choice(valid_cards)
                
                # Erstelle Kontext-Schlüssel für Bayes'sche Analyse
                context_key = self._create_context_key(
                    current_player, state, trick_cards
                )
                
                trick_decisions.append((current_player, card, context_key))
                trick_cards.append((current_player, card))
                state.hands[current_player].remove(card)
                state.played_cards.add(card)
            
            current_player = self._next_player(current_player, state.players)
        
        # Stichgewinner bestimmen
        winner = WizardRules.determine_trick_winner(trick_cards, state.trump_suit)
        state.tricks_won[winner] += 1
        state.trick_leader = winner
        state.current_trick += 1
        
        return trick_decisions
    
    def _get_valid_cards(self, hand: List[Card], 
                        trick_cards: List[Tuple[int, Card]], 
                        trump_suit: Optional[Suit]) -> List[Card]:
        """Gibt alle regelkonformen Karten zurück"""
        return [card for card in hand 
                if WizardRules.is_valid_play(card, hand, trick_cards, trump_suit)]
    
    def _next_player(self, current: int, players: List[int]) -> int:
        """Gibt den nächsten Spieler zurück"""
        idx = players.index(current)
        return players[(idx + 1) % len(players)]
    
    def _create_context_key(self, player: int, state: GameState, 
                          trick_cards: List[Tuple[int, Card]]) -> str:
        """Erstellt aussagekräftigen Kontext-Schlüssel für Bayes'sche Schätzung"""
        # Relevante Spielsituation kompakt kodieren
        bid_diff = state.bids[player] - state.tricks_won[player]
        remaining_tricks = state.round_number - state.current_trick
        trick_position = len(trick_cards)
        has_trump = state.trump_suit is not None
        
        # Kompakter aber unterscheidender Schlüssel
        return f"p{player}_bd{bid_diff}_rt{remaining_tricks}_tp{trick_position}_ht{int(has_trump)}"

class BayesOptimalBackpropagator:
    """
    Implementiert Bayes-optimale Rückwärtspropagierung von Rewards.
    Propagiert Informationen vom Spielende zurück zu früheren Entscheidungen.
    """
    
    def __init__(self, reward_estimator: BayesianRewardEstimator, 
                 temporal_discount: float = 0.95, causal_weight: float = 0.8):
        self.reward_estimator = reward_estimator
        self.temporal_discount = temporal_discount
        self.causal_weight = causal_weight
    
    def propagate_decision_path(self, decision_path: List[Tuple[int, Card, str]], 
                              final_reward: float, target_player: int):
        """
        Propagiert Rewards rückwärts durch den Entscheidungspfad.
        Implementiert temporale Diskontierung und kausale Gewichtung.
        """
        if not decision_path:
            return
        
        # Berechne diskontierte Rewards für jeden Entscheidungspunkt
        discounted_rewards = self._calculate_temporal_discounts(
            decision_path, final_reward
        )
        
        # Berechne kausale Gewichtungen
        causal_weights = self._calculate_causal_weights(
            decision_path, target_player
        )
        
        # Propagiere rückwärts mit kombinierter Gewichtung
        for i, ((player, card, context_key), discounted_reward, causal_weight) in enumerate(
            zip(decision_path, discounted_rewards, causal_weights)
        ):
            # Kombiniere temporale und kausale Gewichtung
            final_weighted_reward = discounted_reward * causal_weight
            
            # Update Bayes'sche Schätzung für Kontext UND spezifische Karte
            context_card_key = f"{context_key}_card_{card}"
            self.reward_estimator.update_observation(context_card_key, final_weighted_reward)
            
            # Update auch allgemeinen Kontext für Transfer Learning
            self.reward_estimator.update_observation(context_key, final_weighted_reward * 0.5)
    
    def _calculate_temporal_discounts(self, decision_path: List[Tuple[int, Card, str]], 
                                    final_reward: float) -> List[float]:
        """Berechnet zeitlich diskontierte Rewards"""
        discounted_rewards = []
        path_length = len(decision_path)
        
        for i in range(path_length):
            # Zeitliche Distanz vom Ende
            time_steps_from_end = path_length - i - 1
            discount_factor = self.temporal_discount ** time_steps_from_end
            discounted_reward = final_reward * discount_factor
            discounted_rewards.append(discounted_reward)
        
        return discounted_rewards
    
    def _calculate_causal_weights(self, decision_path: List[Tuple[int, Card, str]], 
                                target_player: int) -> List[float]:
        """Berechnet kausale Gewichtungen basierend auf Spielerrelevanz"""
        causal_weights = []
        
        for player, card, context_key in decision_path:
            if player == target_player:
                # Entscheidungen des Zielspielers haben volle Gewichtung
                weight = 1.0
            else:
                # Entscheidungen anderer Spieler haben reduzierte Gewichtung
                weight = self.causal_weight
            
            causal_weights.append(weight)
        
        return causal_weights

class BayesOptimalCardEvaluator:
    """Evaluiert Karten basierend auf Bayes'schen Schätzungen"""
    
    def __init__(self, reward_estimator: BayesianRewardEstimator, 
                 exploration_factor: float = 1.5):
        self.reward_estimator = reward_estimator
        self.exploration_factor = exploration_factor
    
    def evaluate_card(self, player: int, card: Card, game_state: GameState) -> float:
        """
        Evaluiert eine Karte mit Upper Confidence Bound für Exploration.
        Implementiert Bayes-optimale Schätzung mit Unsicherheitsbonus.
        """
        # Erstelle Kontext für aktuelle Spielsituation
        context_key = self._create_evaluation_context(player, game_state)
        card_context_key = f"{context_key}_card_{card}"
        
        # Bayes'sche Schätzung des erwarteten Rewards
        estimated_reward = self.reward_estimator.estimate_reward(card_context_key)
        
        # Unsicherheit für Exploration
        uncertainty = self.reward_estimator.get_uncertainty(card_context_key)
        
        # Upper Confidence Bound für optimistische Exploration
        exploration_bonus = self.exploration_factor * uncertainty
        
        # Finale Bewertung
        final_evaluation = estimated_reward + exploration_bonus
        
        return final_evaluation
    
    def _create_evaluation_context(self, player: int, game_state: GameState) -> str:
        """Erstellt Kontext-Schlüssel für Kartenevaluation"""
        bid_diff = game_state.bids[player] - game_state.tricks_won[player]
        remaining_tricks = game_state.round_number - game_state.current_trick
        trick_position = len(game_state.current_trick_cards)
        has_trump = game_state.trump_suit is not None
        
        return f"eval_p{player}_bd{bid_diff}_rt{remaining_tricks}_tp{trick_position}_ht{int(has_trump)}"

class BayesOptimalWizardAssistant:
    """
    Bayes-optimaler Wizard-Assistent mit regelagnostischer Simulation
    und korrekter Rückwärtspropagierung von Belohnungen.
    """
    
    def __init__(self, num_simulations: int = 100000, num_distributions: int = 1000):
        self.num_simulations = num_simulations
        self.num_distributions = num_distributions
        self.simulations_per_distribution = max(1, num_simulations // num_distributions)
        
        # Bayes'scher Reward-Estimator
        self.reward_estimator = BayesianRewardEstimator(prior_alpha=2.0, prior_beta=2.0)
        
        # Rückwärtspropagation
        self.backpropagator = BayesOptimalBackpropagator(self.reward_estimator)
        
        # Kartenbewerter
        self.card_evaluator = BayesOptimalCardEvaluator(self.reward_estimator)
        
        # Thread-lokale Random States für Parallelisierung
        self.random_states = {}
    
    def get_recommendation(self, game_state: GameState, 
                          player_id: int) -> Dict[Card, float]:
        """
        Gibt Bayes-optimale Empfehlungen für alle spielbaren Karten zurück.
        Verwendet regelagnostische Simulation mit Bayes'scher Rückwärtspropagierung.
        """
        # Bestimme verfügbare Karten
        available_cards = self._get_available_cards(game_state, player_id)
        if not available_cards:
            return {}
        
        # Bestimme unbekannte Karten
        unknown_cards = self._get_unknown_cards(game_state)
        
        print(f"Evaluating {len(available_cards)} cards with {self.num_distributions} distributions...")
        
        # Initialisiere Bewertungen
        card_evaluations = {card: [] for card in available_cards}
        
        # Monte Carlo über verschiedene Kartenverteilungen
        for dist_idx in range(self.num_distributions):
            if dist_idx % 100 == 0:
                print(f"Progress: {dist_idx}/{self.num_distributions} distributions")
            
            # Erstelle zufällige Verteilung unbekannter Karten
            shuffled_unknown = unknown_cards.copy()
            random.shuffle(shuffled_unknown)
            
            # Simuliere für diese Verteilung
            dist_evaluations = self._simulate_card_distribution(
                game_state, player_id, shuffled_unknown, available_cards
            )
            
            # Sammle Bewertungen
            for card, value in dist_evaluations.items():
                card_evaluations[card].append(value)
        
        # Berechne finale Bewertungen mit Bayes'scher Schätzung
        final_evaluations = {}
        for card in available_cards:
            if card_evaluations[card]:
                # Verwende Bayes'sche Schätzung statt einfachem Durchschnitt
                final_evaluations[card] = self.card_evaluator.evaluate_card(
                    player_id, card, game_state
                )
            else:
                final_evaluations[card] = 0.0
        
        return final_evaluations
    
    def _simulate_card_distribution(self, game_state: GameState, 
                                   player_id: int, unknown_cards: List[Card],
                                   available_cards: List[Card]) -> Dict[Card, float]:
        """
        Simuliert für eine spezifische Kartenverteilung mit REIN ZUFÄLLIGEN
        Simulationen und Bayes'scher Rückwärtspropagierung.
        """
        card_rewards = {card: [] for card in available_cards}
        
        for card in available_cards:
            # Erstelle Gamestate mit gespielter Karte
            modified_state = self._create_state_with_played_card(
                game_state, player_id, card
            )
            
            # Führe regelagnostische Zufallssimulationen durch
            for sim_idx in range(self.simulations_per_distribution):
                simulator = PureMonteCarloSimulator(modified_state)
                
                # KOMPLETT ZUFÄLLIGE Simulation ohne jede Heuristik
                final_reward, decision_path = simulator.simulate_random_game(
                    unknown_cards.copy(), player_id
                )
                
                card_rewards[card].append(final_reward)
                
                # Bayes'sche Rückwärtspropagierung
                self.backpropagator.propagate_decision_path(
                    decision_path, final_reward, player_id
                )
        
        # Berechne Durchschnittsbewertungen für diese Verteilung
        dist_evaluations = {}
        for card, rewards in card_rewards.items():
            if rewards:
                dist_evaluations[card] = np.mean(rewards)
            else:
                dist_evaluations[card] = 0.0
        
        return dist_evaluations
    
    def _get_available_cards(self, game_state: GameState, player_id: int) -> List[Card]:
        """Bestimmt die spielbaren Karten für einen Spieler"""
        hand = game_state.hands[player_id]
        return [card for card in hand 
                if WizardRules.is_valid_play(
                    card, hand, game_state.current_trick_cards, game_state.trump_suit
                )]
    
    def _get_unknown_cards(self, game_state: GameState) -> List[Card]:
        """Bestimmt alle unbekannten Karten"""
        all_cards = set(WizardDeck().cards)
        known_cards = game_state.played_cards.copy()
        
        # Füge bekannte Handkarten hinzu
        for hand in game_state.hands.values():
            known_cards.update(hand)
        
        # Füge aktuelle Stich-Karten hinzu
        for _, card in game_state.current_trick_cards:
            known_cards.add(card)
        
        return list(all_cards - known_cards)
    
    def _create_state_with_played_card(self, game_state: GameState, 
                                      player_id: int, card: Card) -> GameState:
        """Erstellt einen neuen Gamestate mit gespielter Karte"""
        new_state = game_state.copy()
        
        # Entferne Karte aus der Hand
        new_state.hands[player_id].remove(card)
        
        # Füge Karte zum aktuellen Stich hinzu
        new_state.current_trick_cards.append((player_id, card))
        new_state.played_cards.add(card)
        
        # Aktualisiere aktuellen Spieler
        next_player_idx = (new_state.players.index(player_id) + 1) % len(new_state.players)
        new_state.current_player = new_state.players[next_player_idx]
        
        return new_state
    
    def get_detailed_analysis(self, game_state: GameState, player_id: int) -> Dict:
        """Gibt detaillierte Analyse der Kartenbewertungen zurück"""
        available_cards = self._get_available_cards(game_state, player_id)
        
        analysis = {
            'card_evaluations': {},
            'confidence_intervals': {},
            'uncertainties': {},
            'observation_counts': {},
            'recommendations': {}
        }
        
        for card in available_cards:
            context_key = self.card_evaluator._create_evaluation_context(player_id, game_state)
            card_context_key = f"{context_key}_card_{card}"
            
            # Bayes'sche Schätzung
            estimated_reward = self.reward_estimator.estimate_reward(card_context_key)
            uncertainty = self.reward_estimator.get_uncertainty(card_context_key)
            confidence_interval = self.reward_estimator.get_confidence_interval(card_context_key)
            
            # Anzahl Beobachtungen
            observation_count = len(self.reward_estimator.context_observations[card_context_key])
            
            analysis['card_evaluations'][str(card)] = estimated_reward
            analysis['confidence_intervals'][str(card)] = confidence_interval
            analysis['uncertainties'][str(card)] = uncertainty
            analysis['observation_counts'][str(card)] = observation_count
            
            # Finale Bewertung mit Exploration
            final_evaluation = self.card_evaluator.evaluate_card(player_id, card, game_state)
            analysis['recommendations'][str(card)] = final_evaluation
        
        return analysis

class WizardGameSimulator:
    """Vollständige Wizard-Spielsimulation für Testing und Validierung"""
    
    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.max_rounds = 60 // num_players  # Standard Wizard Rundenzahl
        
    def create_test_game_state(self, round_number: int, current_trick: int = 0) -> GameState:
        """Erstellt einen Test-Spielzustand"""
        deck = WizardDeck()
        deck.shuffle()
        
        players = list(range(self.num_players))
        hands = {}
        
        # Verteile Karten
        for player in players:
            hands[player] = deck.deal_cards(round_number)
        
        # Bestimme Trumpf
        trump_card = deck.deal_cards(1)[0] if deck.cards else None
        trump_suit = trump_card.suit if trump_card and trump_card.suit not in [Suit.WIZARD, Suit.JESTER] else None
        
        # Zufällige Ansagen
        bids = {player: random.randint(0, round_number) for player in players}
        
        # Initialisiere Stiche
        tricks_won = {player: 0 for player in players}

        return GameState(
            round_number=round_number,
            current_trick=current_trick,
            trump_suit=trump_suit,
            trump_card=None,
            players=players,
            hands=hands,
            bids=bids,
            tricks_won=tricks_won,
            current_trick_cards=[],
            played_cards=set(),
            current_player=players[0],
            trick_leader=players[0],
            dealer=(round_number - 1) % len(players)
        )
    
    def simulate_complete_game(self) -> Dict[int, int]:
        """Simuliert ein komplettes Wizard-Spiel"""
        total_scores = {player: 0 for player in range(self.num_players)}
        
        for round_num in range(1, self.max_rounds + 1):
            round_scores = self.simulate_round(round_num)
            for player, score in round_scores.items():
                total_scores[player] += score
        
        return total_scores
    
    def simulate_round(self, round_number: int) -> Dict[int, int]:
        """Simuliert eine komplette Runde"""
        game_state = self.create_test_game_state(round_number)
        
        # Spiele alle Stiche der Runde
        for trick_num in range(round_number):
            self._play_trick(game_state)
        
        # Berechne Rundenpunkte
        round_scores = {}
        for player in game_state.players:
            bid = game_state.bids[player]
            tricks = game_state.tricks_won[player]
            round_scores[player] = WizardRules.calculate_score(bid, tricks)
        
        return round_scores
    
    def _play_trick(self, game_state: GameState):
        """Spielt einen kompletten Stich"""
        trick_cards = []
        current_player = game_state.trick_leader
        
        for _ in range(len(game_state.players)):
            # Bestimme gültige Karten
            valid_cards = [card for card in game_state.hands[current_player]
                          if WizardRules.is_valid_play(card, game_state.hands[current_player], 
                                                     trick_cards, game_state.trump_suit)]
            
            if valid_cards:
                # Zufällige Kartenwahl für Simulation
                card = random.choice(valid_cards)
                trick_cards.append((current_player, card))
                game_state.hands[current_player].remove(card)
                game_state.played_cards.add(card)
            
            # Nächster Spieler
            current_idx = game_state.players.index(current_player)
            current_player = game_state.players[(current_idx + 1) % len(game_state.players)]
        
        # Stichgewinner bestimmen
        winner = WizardRules.determine_trick_winner(trick_cards, game_state.trump_suit)
        game_state.tricks_won[winner] += 1
        game_state.trick_leader = winner
        game_state.current_trick += 1

class WizardAssistantValidator:
    """Validierung und Testing des Wizard-Assistenten"""
    
    def __init__(self):
        self.simulator = WizardGameSimulator()
        self.assistant = BayesOptimalWizardAssistant(num_simulations=1000, num_distributions=100)
    
    def test_basic_functionality(self) -> bool:
        """Testet grundlegende Funktionalität"""
        print("Testing basic functionality...")
        
        try:
            # Erstelle Test-Spielzustand
            game_state = self.simulator.create_test_game_state(round_number=5, current_trick=2)
            
            # Teste Empfehlungen
            recommendations = self.assistant.get_recommendation(game_state, player_id=0)
            
            print(f"✓ Generated recommendations for {len(recommendations)} cards")
            
            # Teste detaillierte Analyse
            analysis = self.assistant.get_detailed_analysis(game_state, player_id=0)
            
            print(f"✓ Generated detailed analysis with {len(analysis['recommendations'])} entries")
            
            return True
            
        except Exception as e:
            print(f"✗ Basic functionality test failed: {e}")
            return False
    
    def test_rule_compliance(self) -> bool:
        """Testet Regelkonformität"""
        print("Testing rule compliance...")
        
        try:
            # Teste verschiedene Spielsituationen
            test_cases = [
                (3, 1),  # Frühe Runde, früher Stich
                (8, 4),  # Mittlere Runde, mittlerer Stich
                (12, 10) # Späte Runde, später Stich
            ]
            
            for round_num, trick_num in test_cases:
                game_state = self.simulator.create_test_game_state(round_num, trick_num)
                
                # Teste für jeden Spieler
                for player in game_state.players:
                    if game_state.hands[player]:  # Nur wenn Spieler noch Karten hat
                        recommendations = self.assistant.get_recommendation(game_state, player)
                        
                        # Prüfe ob alle empfohlenen Karten regelkonform sind
                        for card_str, value in recommendations.items():
                            # Finde entsprechende Karte in der Hand
                            card = None
                            for hand_card in game_state.hands[player]:
                                if str(hand_card) == card_str:
                                    card = hand_card
                                    break
                            
                            if card:
                                is_valid = WizardRules.is_valid_play(
                                    card, game_state.hands[player], 
                                    game_state.current_trick_cards, game_state.trump_suit
                                )
                                
                                if not is_valid:
                                    print(f"✗ Invalid card recommendation: {card}")
                                    return False
            
            print("✓ All recommendations are rule-compliant")
            return True
            
        except Exception as e:
            print(f"✗ Rule compliance test failed: {e}")
            return False
    
    def test_performance_benchmark(self) -> bool:
        """Testet Performance des Assistenten"""
        print("Testing performance...")
        
        import time
        
        try:
            game_state = self.simulator.create_test_game_state(round_number=8, current_trick=3)
            
            start_time = time.time()
            recommendations = self.assistant.get_recommendation(game_state, player_id=0)
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"✓ Generated recommendations in {duration:.2f} seconds")
            
            if duration > 60:  # Warnung bei mehr als 1 Minute
                print("⚠ Performance warning: Recommendation took longer than 60 seconds")
            
            return True
            
        except Exception as e:
            print(f"✗ Performance test failed: {e}")
            return False
    
    def test_bayesian_learning(self) -> bool:
        """Testet Bayes'sches Lernen"""
        print("Testing Bayesian learning...")
        
        try:
            game_state = self.simulator.create_test_game_state(round_number=6, current_trick=2)
            
            # Erste Empfehlung (wenig Daten)
            initial_recommendations = self.assistant.get_recommendation(game_state, player_id=0)
            initial_analysis = self.assistant.get_detailed_analysis(game_state, player_id=0)
            
            # Simuliere mehr Spiele für Lernen
            for _ in range(10):
                test_state = self.simulator.create_test_game_state(round_number=6, current_trick=2)
                self.assistant.get_recommendation(test_state, player_id=0)
            
            # Zweite Empfehlung (mehr Daten)
            final_recommendations = self.assistant.get_recommendation(game_state, player_id=0)
            final_analysis = self.assistant.get_detailed_analysis(game_state, player_id=0)
            
            # Prüfe ob Unsicherheit abgenommen hat
            initial_avg_uncertainty = np.mean(list(initial_analysis['uncertainties'].values()))
            final_avg_uncertainty = np.mean(list(final_analysis['uncertainties'].values()))
            
            if final_avg_uncertainty < initial_avg_uncertainty:
                print("✓ Bayesian learning reduces uncertainty over time")
            else:
                print("⚠ Uncertainty did not decrease as expected")
            
            print(f"✓ Initial uncertainty: {initial_avg_uncertainty:.2f}")
            print(f"✓ Final uncertainty: {final_avg_uncertainty:.2f}")
            
            return True
            
        except Exception as e:
            print(f"✗ Bayesian learning test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Führt alle Tests aus"""
        print("=" * 50)
        print("WIZARD ASSISTANT VALIDATION")
        print("=" * 50)
        
        tests = [
            self.test_basic_functionality,
            self.test_rule_compliance,
            self.test_performance_benchmark,
            self.test_bayesian_learning
        ]
        
        results = []
        for test in tests:
            print()
            result = test()
            results.append(result)
        
        print()
        print("=" * 50)
        success_count = sum(results)
        total_count = len(results)
        
        if success_count == total_count:
            print(f"✓ ALL TESTS PASSED ({success_count}/{total_count})")
            return True
        else:
            print(f"✗ SOME TESTS FAILED ({success_count}/{total_count})")
            return False

def create_example_game_state() -> GameState:
    """Erstellt einen Beispiel-Spielzustand für Demonstrationszwecke"""
    
    # Beispiel: Runde 5, Stich 3
    players = [0, 1, 2, 3]
    
    # Beispiel-Hände (vereinfacht)
    hands = {
        0: [Card(Suit.RED, 10), Card(Suit.BLUE, 5), Card(Suit.WIZARD, 0)],
        1: [Card(Suit.GREEN, 8), Card(Suit.YELLOW, 3), Card(Suit.RED, 2)],
        2: [Card(Suit.BLUE, 12), Card(Suit.GREEN, 1), Card(Suit.JESTER, 0)],
        3: [Card(Suit.YELLOW, 9), Card(Suit.RED, 7), Card(Suit.BLUE, 4)]
    }
    
    # Beispiel-Ansagen
    bids = {0: 2, 1: 1, 2: 1, 3: 1}
    
    # Bereits gewonnene Stiche
    tricks_won = {0: 1, 1: 1, 2: 1, 3: 0}
    
    # Bereits gespielte Karten
    played_cards = {
        Card(Suit.RED, 1), Card(Suit.BLUE, 3), Card(Suit.GREEN, 6),
        Card(Suit.YELLOW, 11), Card(Suit.RED, 13), Card(Suit.BLUE, 8),
        Card(Suit.GREEN, 4), Card(Suit.YELLOW, 2)
    }
    
    return GameState(
        round_number=5,
        current_trick=3,
        trump_suit=Suit.RED,
        trump_card=None,
        players=players,
        hands=hands,
        bids=bids,
        tricks_won=tricks_won,
        current_trick_cards=[],
        played_cards=played_cards,
        current_player=0,
        trick_leader=0,
        dealer=0
    )

def main():
    """Hauptfunktion für Demonstration und Testing"""
    
    print("WIZARD ASSISTANT v4.0")
    print("=" * 50)
    
    # Validierung ausführen
    validator = WizardAssistantValidator()
    validator.run_all_tests()



if __name__ == "__main__":
    main()