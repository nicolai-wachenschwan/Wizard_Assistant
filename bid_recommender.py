### Bidding Simulation ###
import numpy as np
import pandas as pd
from typing import Dict, List
from wizard_core_game_classes import *
from simulator import *

class BayesOptimalBidRecommender:
    """
    Simuliert mögliche Gebote (Bids) für einen Spieler in einem gegebenen Spielzustand
    und gibt die Ergebnisse als Pandas DataFrame zurück.
    """
    def __init__(self, num_simulations: int = 500):
        self.num_simulations = num_simulations

    def recommend_bid(
        self,
        game_state: GameState,
        player_id: int
    ) -> pd.DataFrame:
        """
        Simuliert für alle möglichen Gebote (0 bis Anzahl Karten in der Hand)
        und gibt einen sortierten DataFrame mit Metriken (Mittelwert, Varianz,
        95% Konfidenzintervall-Obergrenze) für jedes Gebot zurück.
        """
        max_bid = len(game_state.hands.get(player_id, []))
        if max_bid == 0:
            return pd.DataFrame() # Leeren DataFrame zurückgeben

        unknown_cards = self._get_unknown_cards(game_state, player_id)
        bid_rewards: Dict[int, List[float]] = {bid: [] for bid in range(max_bid + 1)}

        for _ in range(self.num_simulations):
                # 1. Zufällige Verteilung der unbekannten Karten (mit dem oben genannten Bugfix)
                simulated_full_state = self._distribute_unknown_cards(game_state, player_id, unknown_cards)
                
                # NEU: Identifiziere Spieler, die nach player_id bieten
                player_order = game_state.players
                my_turn_index = player_order.index(player_id)
                subsequent_bidders = player_order[my_turn_index + 1:]

                # NEU: Simuliere die Gebote der nachfolgenden Spieler
                # Dies ist eine einfache zufällige Annahme. Eine bessere Logik könnte
                # die Hand des simulierten Spielers analysieren.
                for bidder_id in subsequent_bidders:
                    # Simuliere ein zufälliges Gebot für diesen Spieler
                    simulated_full_state.bids[bidder_id] = np.random.randint(0, max_bid + 1)

                for bid in range(max_bid + 1):
                    # Setze das Test-Gebot für den Hauptspieler
                    test_state = simulated_full_state.copy()
                    test_state.bids[player_id] = bid
                    
                    # Simuliere das Spiel mit diesem vollständigen Zustand
                    reward = self._simulate_round(test_state, player_id, unknown_cards)
                    bid_rewards[bid].append(reward)

        # Liste zur Sammlung der Ergebnisse für jedes Gebot
        results_list = []

        for bid, rewards in bid_rewards.items():
            if not rewards:
                continue

            mean = np.mean(rewards)
            variance = np.var(rewards, ddof=1) if len(rewards) > 1 else 0 # Stichprobenvarianz
            std_dev = np.sqrt(variance)
            
            # Berechnung der oberen Grenze des 95% Konfidenzintervalls
            # Z-Wert für 95% ist 1.96
            confidence_upper_bound = mean + 1.96 * std_dev

            results_list.append({
                'Gebot': bid,
                'Mittelwert': mean,
                'Varianz': variance,
                '95% Konfidenz Oben': confidence_upper_bound
            })
        
        if not results_list:
            return pd.DataFrame()

        # DataFrame aus der Liste erstellen
        results_df = pd.DataFrame(results_list)
        
        # DataFrame nach der vielversprechendsten Option sortieren
        results_df = results_df.sort_values('95% Konfidenz Oben', ascending=False).reset_index(drop=True)
        
        return results_df
    def _distribute_unknown_cards(self, game_state: 'GameState', player_id: int, unknown_cards: List['Card']) -> 'GameState':
        """
        Verteilt die unbekannten Karten zufällig auf die anderen Spieler und erstellt
        einen vollständigen Spielzustand für die Simulation.
        """
        new_state = game_state.copy()
        remaining_cards = unknown_cards.copy()
        np.random.shuffle(remaining_cards)
        
        card_index = 0
        # Verteile Karten an alle anderen Spieler
        for pid in new_state.hands:
            if pid != player_id:
                # BUGFIX: Hand des Gegners explizit leeren, um Informationsleck zu schließen
                new_state.hands[pid] = [] 
                
                expected_hand_size = len(game_state.hands.get(player_id, []))
                
                cards_needed = expected_hand_size # Da die Hand nun leer ist
                if cards_needed > 0 and card_index + cards_needed <= len(remaining_cards):
                    new_state.hands[pid].extend(remaining_cards[card_index:card_index + cards_needed])
                    card_index += cards_needed
        
        return new_state    

    # Die restlichen Methoden (_simulate_bid, _simulate_round, _get_unknown_cards) bleiben unverändert
    def _simulate_bid(self, game_state: 'GameState', player_id: int, bid: int, unknown_cards: List['Card']) -> 'GameState':
        new_state = game_state.copy()
        new_state.bids[player_id] = bid
        return new_state

    def _simulate_round(self, game_state: 'GameState', player_id: int, unknown_cards: List['Card']) -> float:
        simulator = PureMonteCarloSimulator(game_state)
        sim_bids, sim_tricks_won, _ = simulator.simulate_random_game(unknown_cards)
        return WizardRules.calculate_score(sim_bids.get(player_id, 0), sim_tricks_won.get(player_id, 0))

    def _get_unknown_cards(self, game_state: 'GameState', player_id: int) -> List['Card']:
        all_deck = set(WizardDeck().cards)
        known = game_state.played_cards.copy()
        for _, c in game_state.current_trick_cards:
            known.add(c)
        own_hand = set(game_state.hands.get(player_id, []))
        known_to_player = known.union(own_hand)
        return list(all_deck - known_to_player)


