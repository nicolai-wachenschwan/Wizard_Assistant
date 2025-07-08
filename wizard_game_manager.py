# wizard_game_manager.py

from typing import Dict, List, Optional, Tuple
from wizard_core_game_classes import GameState, WizardDeck, Suit, Card, WizardRules

class WizardGameManager:
    """
    Verwaltet den Zustand und die Logik eines Wizard-Spiels.
    Diese Klasse ist unabhängig von der Benutzeroberfläche (GUI).
    """
    def __init__(self):
        # Initialzustand, wird beim Start eines neuen Spiels gesetzt
        self.game_state: Optional[GameState] = None
        self.player_types: Dict[int, str] = {}
        self.total_scores: Dict[int, int] = {}
        self.current_round_scores: Dict[int, int] = {}
        self.human_player_id: int = 0
        self.game_phase: str = "setup"  # Phasen: setup, bidding, play_game, round_over
        self.last_trick: Optional[Dict] = None
        # Add new attributes for game options
        self.deal_digitally: bool = True
        self.confirm_play: bool = False
        self.last_wizard_wins: bool = False
        self.wizard_trump_chooser: Optional[int] = None

    def start_new_game(self, num_players: int, player_types: Dict[int, str], human_player_id: int, start_round: int = 1, deal_digitally: bool = True, confirm_play: bool = False,last_wizard_wins: bool = False, dealer: Optional[int] = None):
        """Initialisiert ein komplett neues Spiel."""
        self.player_types = player_types
        self.human_player_id = human_player_id
        self.deal_digitally = deal_digitally
        self.confirm_play = confirm_play
        self.last_wizard_wins = last_wizard_wins
        self.total_scores = {p: 0 for p in range(num_players)}
        self.start_new_round(round_number=start_round, num_players=num_players, dealer=dealer)

    def is_human_hand_set(self):
        """Check if the human player's hand has been set (for manual card input)."""
        if not self.game_state:
            return True
        
        # Wenn Karten digital verteilt werden, ist die Hand automatisch gesetzt
        if self.deal_digitally:
            return True
        
        # Wenn Karten analog verteilt werden, prüfe ob die Hand manuell eingegeben wurde
        human_hand = self.game_state.hands.get(self.human_player_id, [])
        expected_cards = self.game_state.round_number
        return len(human_hand) == expected_cards

    def set_player_hand(self, player_id: int, hand_cards: List[Card]):
        """Set the hand cards for a specific player (used for manual input)."""
        if self.game_state:
            self.game_state.hands[player_id] = hand_cards


    def start_new_round(self, round_number: int, num_players: int, dealer: Optional[int] = None):
        """Bereitet den Spielzustand für eine neue Runde vor."""
        self.last_trick = None
        players = list(range(num_players))
        if dealer is None:
            dealer = (round_number - 1) % num_players
        start_player = (dealer + 1) % num_players

        deck = WizardDeck()
        deck.shuffle()

        if self.deal_digitally:
            hands = {p: sorted(deck.deal_cards(round_number)) for p in players}
            trump_card = None
            if sum(len(h) for h in hands.values()) < 60:
                trump_card = deck.deal_cards(1)[0]
        else:
            hands = {p: [] for p in players}
            trump_card = None

        played_cards = set()
        if trump_card:
            played_cards.add(trump_card)

        self.game_state = GameState(
            round_number=round_number, current_trick=0, trump_suit=None,
            trump_card=trump_card,
            players=players, hands=hands, bids={}, tricks_won={p: 0 for p in players},
            current_trick_cards=[], played_cards=played_cards,
            current_player=start_player, trick_leader=start_player,
            dealer=dealer,
            last_wizard_wins=self.last_wizard_wins  # NEUES ARGUMENT
        )
        self.current_round_scores = {p: 0 for p in players}
        self.game_phase = "bidding"

        if trump_card:
            self.set_trump_from_card(trump_card, chooser=start_player)

    # NEU: Methode zum manuellen Setzen der Trumpffarbe
    def set_trump_suit_manually(self, suit_value: Optional[str]):
        """Sets the trump suit from a string value, used for manual input."""
        if self.game_state:
            if suit_value is None or suit_value == "Keine":
                self.game_state.trump_suit = None
            else:
                self.game_state.trump_suit = Suit(suit_value)

    def set_bids(self, bids: Dict[int, int]):
        """Speichert die Gebote und startet die Spielphase."""
        if self.game_state:
            self.game_state.bids = bids
            self.game_phase = "play_game"
            
    def set_trump_from_card(self, trump_card: Optional[Card], chooser: Optional[int] = None):
        """Set the trump suit based on the trump card."""
        if not self.game_state:
            return

        self.game_state.trump_card = trump_card
        self.wizard_trump_chooser = None

        if not trump_card:
            self.game_state.trump_suit = None
            return

        if trump_card.suit == Suit.JESTER:
            self.game_state.trump_suit = None
        elif trump_card.suit == Suit.WIZARD:
            if chooser is None:
                chooser = (self.game_state.dealer + 1) % len(self.game_state.players)
            if self.player_types.get(chooser) == "computer":
                counts = {s: 0 for s in [Suit.RED, Suit.BLUE, Suit.GREEN, Suit.YELLOW]}
                for c in self.game_state.hands.get(chooser, []):
                    if c.suit in counts:
                        counts[c.suit] += 1
                preferred = max(counts, key=counts.get)
                self.game_state.trump_suit = preferred
            else:
                self.game_state.trump_suit = None
                self.wizard_trump_chooser = chooser
        else:
            self.game_state.trump_suit = trump_card.suit

    def choose_trump_suit(self, suit_value: Optional[str]):
        """Finalize trump suit selection when a Wizard was revealed."""
        if not self.game_state or self.wizard_trump_chooser is None:
            return
        if suit_value is None or suit_value == "Keine":
            self.game_state.trump_suit = None
        else:
            self.game_state.trump_suit = Suit(suit_value)
        self.wizard_trump_chooser = None
            
    def play_card(self, player_id: int, card: Card):
        """
        Verarbeitet das Ausspielen einer Karte, aktualisiert den Spielzustand
        und vervollständigt ggf. den Stich oder die Runde.
        """
        if not self.game_state:
            return

        # Karte aus der Hand entfernen (falls es keine manuell eingegebene Karte ist)
        if card in self.game_state.hands[player_id]:
            self.game_state.hands[player_id].remove(card)
            
        self.game_state.current_trick_cards.append((player_id, card))
        self.game_state.played_cards.add(card)
        
        # Nächsten Spieler bestimmen
        current_idx = self.game_state.players.index(player_id)
        next_player_idx = (current_idx + 1) % len(self.game_state.players)
        self.game_state.current_player = self.game_state.players[next_player_idx]
        
        # Prüfen, ob der Stich beendet ist
        if len(self.game_state.current_trick_cards) == len(self.game_state.players):
            self._complete_trick()

    def _complete_trick(self):
        """Vervollständigt einen Stich, bestimmt den Gewinner und prüft auf Rundenende."""
        if not self.game_state:
            return
            
        winner = WizardRules.determine_trick_winner(
            self.game_state.current_trick_cards, 
            self.game_state.trump_suit,
            self.game_state.last_wizard_wins # NEUES ARGUMENT
        )
        self._save_last_trick(self.game_state.current_trick_cards, winner)
        
        self.game_state.tricks_won[winner] += 1
        self.game_state.current_trick += 1
        self.game_state.current_trick_cards = []
        self.game_state.current_player = winner
        self.game_state.trick_leader = winner
        
        # Prüfen, ob die Runde beendet ist
        if self.game_state.current_trick >= self.game_state.round_number:
            self._complete_round()

    def _complete_round(self):
        """Vervollständigt eine Runde, berechnet die Punkte und setzt die nächste Phase."""
        if not self.game_state:
            return

        for player in self.game_state.players:
            score = WizardRules.calculate_score(
                self.game_state.bids.get(player, 0), self.game_state.tricks_won.get(player, 0)
            )
            self.current_round_scores[player] = score
            self.total_scores[player] += score
        
        self.game_phase = "round_over"

    def _save_last_trick(self, trick_cards: List[Tuple[int, Card]], winner: int):
        """Speichert den letzten Stich zur Anzeige."""
        self.last_trick = {
            'cards': trick_cards.copy(),
            'winner': winner
        }

    def proceed_to_next_round(self):
        """Startet die nächste Runde des Spiels."""
        if self.game_state:
            num_players = len(self.game_state.players)
            next_round_number = self.game_state.round_number + 1
            max_rounds = 60 // num_players
            if next_round_number <= max_rounds:
                new_dealer = (self.game_state.dealer + 1) % num_players
                self.start_new_round(next_round_number, num_players, dealer=new_dealer)
