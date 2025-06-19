from typing import List, Dict, Tuple, Optional
import random
from wizard_core_game_classes import *

class PureMonteCarloSimulator:
    """
    Führe von einem gegebenen Spielzustand an eine rein zufällige Vollsimulation
    durch. Gibt am Ende:
    - bids_final: Dict{Spieler:Bid}
    - tricks_won_final: Dict{Spieler:gewonnene Stiche}
    - decision_path: Liste der Entscheidungen (Spieler, Karte, Kontext, valid_actions)
    """
    def __init__(self, game_state: GameState):
        # initial_state: Kopie des Zustands *nach* dem Zug, den wir gerade untersuchen
        self.initial_state = game_state.copy()
    
    def simulate_random_game(
        self,
        unknown_cards: List[Card]
    ) -> Tuple[Dict[int,int], Dict[int,int], List[Tuple[int, Card, str, List[Card]]]]:
        """
        Führe eine zufällige Fortsetzung des Spiels durch bis zum Ende dieser Runde.
        unknown_cards: Liste aller Karten, die weder in known hands noch auf Tisch sind.
        Rückgabe:
          - sim_bids: Gebote am Ende der Simulation (werden von Simulation nicht verändert, 
            daher einfach nur weitergereicht)
          - sim_tricks_won: tatsächlich gezählte Stiche am Ende
          - decision_path: Liste von Quads (spieler_id, played_card, context_key, valid_actions_list)
        """
        state = self.initial_state.copy()
        # Verteile unbekannte Karten zufällig
        self._distribute_unknown_cards(state, unknown_cards)
        
        decision_path_segment: List[Tuple[int, Card, str, List[Card]]] = []
        
        # 1) Falls der aktuelle Stich schon begonnen hat und noch nicht alle Spieler gespielt haben:
        if 0 < len(state.current_trick_cards) < len(state.players):
            part_decisions = self._simulate_one_trick(state)
            decision_path_segment.extend(part_decisions)
        
        # 2) Solange es im Round noch Stiche gibt:
        while state.current_trick < state.round_number:
            if state.current_trick_cards:
                # Falls aus irgendeinem Grund noch Karten in current_trick_cards sind, löschen
                state.current_trick_cards = []
            part_decisions = self._simulate_one_trick(state)
            decision_path_segment.extend(part_decisions)
        
        return state.bids, state.tricks_won, decision_path_segment
    
    def _distribute_unknown_cards(self, state: GameState, unknown_cards: List[Card]):
        """
        Verteilt die unbekannten Karten (unknown_cards) zufällig auf alle Spieler­hände,
        sodass in state.hands jeder Spieler die korrekte Anzahl Karten für die verbleibenden Stiche hat.
        """
        cards_to_deal = unknown_cards.copy()
        random.shuffle(cards_to_deal)
        idx = 0
        
        for player_id in state.players:
            player_hand = state.hands.setdefault(player_id, [])
            # Berechne, wie viele Karten dieser Spieler jetzt noch haben sollte:
            # Er hat zu Beginn der Runde `round_number` Karten. 
            # Falls current_trick = t, haben alle Spieler t Karten gespielt, 
            # plus evtl. 1 Karte, wenn sie bereits im laufenden Stich gespielt haben.
            cards_at_trick_start = state.round_number - state.current_trick
            has_played = any(p == player_id for p, _ in state.current_trick_cards)
            target_size = cards_at_trick_start - (1 if has_played else 0)
            missing = max(0, target_size - len(player_hand))
            for _ in range(missing):
                if idx < len(cards_to_deal):
                    player_hand.append(cards_to_deal[idx])
                    idx += 1
                else:
                    break
            state.hands[player_id] = player_hand
    
    def _simulate_one_trick(
        self,
        state: GameState
    ) -> List[Tuple[int, Card, str, List[Card]]]:
        """
        Simuliert exakt einen Stich ausgehend von `state.current_trick_cards`.
        Jeder Spieler wählt per `random.choice(valid_actions)` eine Karte, 
        und wir notieren für jeden Zug (spieler, karte, kontext, valid_actions).
        """
        cards_in_this_trick = state.current_trick_cards.copy()
        decisions_made_this_call: List[Tuple[int, Card, str, List[Card]]] = []
        
        num_played = len(cards_in_this_trick)
        num_to_play = len(state.players) - num_played
        current = state.current_player
        
        for _ in range(num_to_play):
            player_hand = state.hands.get(current, [])
            valid_actions = self._get_valid_cards(player_hand, cards_in_this_trick, state.trump_suit)
            
            if not valid_actions:
                # Falls keine gültigen Aktionen (eigentlich ungewöhnlich), skip
                current = self._next_player(current, state.players)
                continue
            
            # Erzeuge Kontext‐Key für diese Entscheidung
            context_key = self._create_context_key(current, state, cards_in_this_trick)
            # WICHTIG: Wir notieren valid_actions, damit wir später im Backprop. wissen,
            # welche Alternativen es gegeben hätte.
            played_card = random.choice(valid_actions)
            decisions_made_this_call.append((current, played_card, context_key, list(valid_actions)))
            
            # Karte ablegen
            cards_in_this_trick.append((current, played_card))
            state.hands[current].remove(played_card)
            state.played_cards.add(played_card)
            current = self._next_player(current, state.players)
        
        # Wer hat den Stich gewonnen?
        if cards_in_this_trick:
            winner = WizardRules.determine_trick_winner(cards_in_this_trick, state.trump_suit)
            if winner != -1:
                state.tricks_won[winner] = state.tricks_won.get(winner, 0) + 1
            state.trick_leader = winner
        
        state.current_player = state.trick_leader
        state.current_trick_cards = []
        state.current_trick += 1
        
        return decisions_made_this_call
    
    def _get_valid_cards(
        self,
        hand: List[Card],
        trick_cards: List[Tuple[int, Card]],
        trump_suit: Optional[Suit]
    ) -> List[Card]:
        if not hand:
            return []
        return [
            c for c in hand 
            if WizardRules.is_valid_play(c, hand, trick_cards, trump_suit)
        ]
    
    def _next_player(self, current: int, players: List[int]) -> int:
        idx = players.index(current)
        return players[(idx + 1) % len(players)]
    
    def _create_context_key(
        self,
        player: int,
        state: GameState,
        trick_cards_in_progress: List[Tuple[int, Card]]
    ) -> str:
        """
        Erzeuge einen kurzen String, der den Kontext beschreibt:
        - p{player}_bd{bid_diff}_rt{remaining_tricks}_tp{trick_pos}_ht{has_trump}
        """
        bid_diff = state.bids.get(player, 0) - state.tricks_won.get(player, 0)
        remaining_tricks = state.round_number - state.current_trick
        trick_position = len(trick_cards_in_progress)
        has_trump = 1 if state.trump_suit is not None else 0
        return f"p{player}_bd{bid_diff}_rt{remaining_tricks}_tp{trick_position}_ht{has_trump}"
