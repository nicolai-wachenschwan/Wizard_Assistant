from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set
import random
import copy

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
    value: int  # 1-13 für normale Karten, 0 für Wizard/Jester
    
    def __post_init__(self):
        if self.suit in [Suit.WIZARD, Suit.JESTER]:
            self.value = 0
    
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
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.value == other.value
    
    def __hash__(self):
        return hash((self.suit, self.value))
    
    def __lt__(self, other):
        """Ermöglicht das Sortieren von Karten, erst nach Farbe, dann nach Wert."""
        if not isinstance(other, Card):
            return NotImplemented

        # Definieren Sie eine feste Reihenfolge für die Farben
        suit_order = {
            Suit.YELLOW: 0,
            Suit.GREEN: 1,
            Suit.BLUE: 2,
            Suit.RED: 3,
            Suit.JESTER: 4,
            Suit.WIZARD: 5
        }
        
        # Wenn die Farben unterschiedlich sind, sortiere nach der Farb-Reihenfolge
        if self.suit != other.suit:
            return suit_order[self.suit] < suit_order[other.suit]
        
        # Wenn die Farben gleich sind, sortiere nach dem Wert
        return self.value < other.value

class WizardDeck:
    """Verwaltet das Wizard-Kartendeck."""
    def __init__(self):
        self.cards = self._create_deck()
    
    def _create_deck(self) -> List[Card]:
        cards: List[Card] = []
        # Normale Karten: 4 Farben × 13 Werte
        for suit in [Suit.RED, Suit.BLUE, Suit.GREEN, Suit.YELLOW]:
            for value in range(1, 14):
                cards.append(Card(suit, value))
        # Spezialkarten: 4 Wizard, 4 Jester
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
    """
    Repräsentiert den aktuellen Spielzustand.
    - round_number: Anzahl Karten pro Hand / Anzahl Stiche in der Runde
    - current_trick: Index des gerade laufenden Stichs (0-basiert)
    - trump_suit: Trumpfarbe (oder None)
    - players: Liste von Spieler‐IDs
    - hands: Dict von Spieler-ID -> Liste seiner Karten
    - bids: Dict von Spieler-ID -> angesagte Stichzahl
    - tricks_won: Dict von Spieler-ID -> bereits gewonnene Stiche
    - current_trick_cards: Liste von (spielerID, Karte) für den laufenden Stich
    - played_cards: Set aller Karten, die in dieser Runde bereits gespielt wurden
    - current_player: welcher Spieler ist gerade am Zug
    - trick_leader: wer hat den letzten Stich gewonnen (führt nächsten an)
    """
    round_number: int
    current_trick: int
    trump_suit: Optional[Suit]
    players: List[int]
    hands: Dict[int, List[Card]]
    bids: Dict[int, int]
    tricks_won: Dict[int, int]
    current_trick_cards: List[Tuple[int, Card]]
    played_cards: Set[Card]
    current_player: int
    trick_leader: int
    last_wizard_wins: bool = False
    
    def copy(self):
        return copy.deepcopy(self)

class WizardRules:
    """Implementiert die grundlegenden Wizard-Spielregeln."""
    
    @staticmethod
    def is_valid_play(
        card: Card,
        hand: List[Card],
        current_trick_cards: List[Tuple[int, Card]],
        trump_suit: Optional[Suit]
    ) -> bool:
        """
        Prüft, ob `card` in der aktuellen Hand `hand` und angesichts des aktuellen Stichs
        (current_trick_cards) und Trumpffarbe trump_suit gespielt werden darf.
        """
        # 1) Wenn erster Spieler im Stich, immer erlaubt
        if not current_trick_cards:
            return True
        
        # 2) Wizard und Jester immer spielen dürfen
        if card.suit in [Suit.WIZARD, Suit.JESTER]:
            return True
        
        # 3) Bestimme die führende Farbe (erste nicht‐Wizard‐/Jester‐Karte)
        lead_suit: Optional[Suit] = None
        for _, trick_card in current_trick_cards:
            if trick_card.suit not in [Suit.WIZARD, Suit.JESTER]:
                lead_suit = trick_card.suit
                break
        
        # Wenn bisher nur Wizard/Jester gespielt wurden, darf man jede Farbe legen
        if lead_suit is None:
            return True
        
        # Wenn die Karte die führende Farbe bedient, ist sie gültig
        if card.suit == lead_suit:
            return True
        
        # Prüfe, ob der Spieler noch eine Karte der führenden Farbe hat (ausgenommen Wizard/Jester)
        can_follow_suit = any(
            (c.suit == lead_suit and c.suit not in [Suit.WIZARD, Suit.JESTER])
            for c in hand
        )
        # Falls er diese Farbe hat, MUSS er sie spielen, andernfalls ist der Schieben‐Zug erlaubt
        return not can_follow_suit
    
    @staticmethod
    def determine_trick_winner(
        trick_cards: List[Tuple[int, Card]],
        trump_suit: Optional[Suit],
        last_wizard_wins: bool = False  # Neuer Parameter für die Regel
    ) -> int:
        """
        Bestimmt den Gewinner des gerade beendeten Stichs `trick_cards`
        (Liste von (SpielerID, Karte)). Rückgabe: SpielerID des Stichgewinners.
        """
        if not trick_cards:
            return -1
        
        # 1) Wizard schlägt immer: Der erste oder letzte gespielte Wizard gewinnt
        wizard_plays = [(p, c) for p, c in trick_cards if c.suit == Suit.WIZARD]
        if wizard_plays:
            if last_wizard_wins:
                return wizard_plays[-1][0]  # Der letzte gespielte Wizard gewinnt
            else:
                return wizard_plays[0][0]   # Der erste gespielte Wizard gewinnt (Standard)
        
        # 2) Bestimme die führende Farbe (erste Nicht‐Jester‐Karte)
        lead_suit: Optional[Suit] = None
        for _, card_in_trick in trick_cards:
            if card_in_trick.suit != Suit.JESTER:
                lead_suit = card_in_trick.suit
                break
        
        # Wenn nur Jester gespielt wurden, gewinnt der erste Jester
        if lead_suit is None:
            return trick_cards[0][0]
        
        best_trump_card: Optional[Card] = None
        trump_winner: int = -1
        best_lead_card: Optional[Card] = None
        lead_winner: int = -1
        
        for player, card_in_trick in trick_cards:
            if card_in_trick.suit == Suit.JESTER:
                continue
            # Trumpf-Logik
            if card_in_trick.suit == trump_suit:
                if best_trump_card is None or card_in_trick.value > best_trump_card.value:
                    best_trump_card = card_in_trick
                    trump_winner = player
            elif card_in_trick.suit == lead_suit:
                if best_lead_card is None or card_in_trick.value > best_lead_card.value:
                    best_lead_card = card_in_trick
                    lead_winner = player
        
        if trump_winner != -1:
            return trump_winner
        elif lead_winner != -1:
            return lead_winner
        else:
            # Fallback: Der erste Nicht‐Jester im Stich (sollte theoretisch nie zutreffen)
            for p, c in trick_cards:
                if c.suit != Suit.JESTER:
                    return p
            return trick_cards[0][0]

    @staticmethod
    def calculate_score(bid: int, tricks_won: int) -> int:
        """
        Berechnet die Punkte am Ende der Runde:
        - Wenn bid == tricks_won: 20 + 10 * bid
        - Sonst: −10 * |bid − tricks_won|
        """
        if bid == tricks_won:
            return 20 + 10 * bid
        else:
            return -10 * abs(bid - tricks_won)
