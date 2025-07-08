# wizard_gui.py

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import copy 

# Kern-Klassen und der neue Game-Manager
from wizard_core_game_classes import Card, Suit, WizardRules
from wizard_game_manager import WizardGameManager

# KI-Komponenten
from wizard_assistant_o3 import BayesOptimalWizardAssistant
from bid_recommender import BayesOptimalBidRecommender

# Streamlit page config
st.set_page_config(
    page_title="Wizard Assistant GUI",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class WizardGUI:
    def __init__(self):
        # Initialisiere den Game Manager im Session State, falls nicht vorhanden
        if 'game_manager' not in st.session_state:
            st.session_state.game_manager = WizardGameManager()
        self.game_manager: WizardGameManager = st.session_state.game_manager

        # Initialisiere KI-Parameter im Session State, falls nicht vorhanden
        self.init_ai_params()
        if 'player_names' not in st.session_state:
            st.session_state.player_names = {}
        
        if 'history' not in st.session_state:
            st.session_state.history = []

        # KI-Komponenten mit Werten aus dem Session State initialisieren
        self.assistant = BayesOptimalWizardAssistant(
            num_simulations=st.session_state.ai_card_sims,
            num_distributions=st.session_state.ai_card_dists
        )
        self.bid_recommender = BayesOptimalBidRecommender(
            num_simulations=st.session_state.ai_bid_sims
        )
    
    def init_ai_params(self):
        """Initialisiert die Standard-KI-Parameter im Session State."""
        if 'ai_card_sims' not in st.session_state:
            st.session_state.ai_card_sims = 500
        if 'ai_card_dists' not in st.session_state:
            st.session_state.ai_card_dists = 50
        if 'ai_bid_sims' not in st.session_state:
            st.session_state.ai_bid_sims = 200

    # --- HILFSMETHODEN (inkl. Undo-Logik) ---

    def save_state_for_undo(self):
        st.session_state.history.append(copy.deepcopy(st.session_state.game_manager))

    def undo_last_action(self):
        if st.session_state.history:
            st.session_state.game_manager = st.session_state.history.pop()
            for key in ['selected_card_to_play', 'bid_recommendations', 'last_recommendations', 'bids_in_progress']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        else:
            st.toast("Keine weiteren Aktionen zum Rückgängigmachen vorhanden.")

    # --- DARSTELLUNGS-METHODEN (unverändert) ---

    def create_card_image(self, card: Card, width: int = 80, height: int = 120) -> Image.Image:
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width-1, height-1], outline='black', width=2)
        color_map = {Suit.RED: 'red', Suit.BLUE: 'blue', Suit.GREEN: 'green', Suit.YELLOW: 'orange', Suit.WIZARD: 'purple', Suit.JESTER: 'black'}
        color = color_map.get(card.suit, 'black')
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        text = "W" if card.suit == Suit.WIZARD else ("J" if card.suit == Suit.JESTER else str(card.value))
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((width - text_width) / 2, (height - text_height) / 2), text, fill=color, font=font)
        return img

    def get_player_name(self, player_id: int) -> str:
        """Returns the custom name for a player if set."""
        names = st.session_state.get('player_names', {})
        return names.get(player_id, f"Spieler {player_id + 1}")
    
    def display_game_overview(self):
        game_state = self.game_manager.game_state
        st.header(f"🎮 Runde {game_state.round_number} / Stich {game_state.current_trick + 1}")
        overview_data = [{
                "Spieler": f"{self.get_player_name(p)} ({'👤' if self.game_manager.player_types.get(p) == 'human' else '🤖'})",
                "Gebot": game_state.bids.get(p, 0), "Stiche": game_state.tricks_won.get(p, 0),
                "Punkte": self.game_manager.current_round_scores.get(p, 0),
                "Gesamt": self.game_manager.total_scores.get(p, 0)
            } for p in game_state.players]
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True, hide_index=True)
        st.metric("Trumpf", game_state.trump_suit.value if game_state.trump_suit else "Keine")

    def display_current_trick(self):
        game_state = self.game_manager.game_state
        if game_state.current_trick_cards:
            st.subheader("🃏 Aktueller Stich")
            trick_cols = st.columns(len(game_state.players))
            for i, (player, card) in enumerate(game_state.current_trick_cards):
                with trick_cols[i]:
                    st.write(f"**{self.get_player_name(player)}**")
                    st.image(self.create_card_image(card), caption=str(card))

    def display_last_trick(self):
        if self.game_manager.last_trick:
            st.subheader("🔄 Letzter Stich")
            last_trick = self.game_manager.last_trick
            trick_cols = st.columns(len(last_trick['cards']))
            for i, (player, card) in enumerate(last_trick['cards']):
                with trick_cols[i]:
                    is_winner = player == last_trick['winner']
                    name = self.get_player_name(player)
                    st.write(f"**{'🏆 ' if is_winner else ''}{name}{' (Gewinner)' if is_winner else ''}**")
                    st.image(self.create_card_image(card), caption=str(card))

    def display_round_results(self):
        game_state = self.game_manager.game_state
        st.header(f"📊 Ergebnis von Runde {game_state.round_number}")
        results_data = [{
            "Spieler": self.get_player_name(p), "Gebot": game_state.bids.get(p, 0),
            "Stiche": game_state.tricks_won.get(p, 0),
            "Rundenpunkte": self.game_manager.current_round_scores.get(p, 0),
            "Gesamtpunkte": self.game_manager.total_scores.get(p, 0)
        } for p in game_state.players]
        st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)

        max_rounds = 60 // len(game_state.players)
        if game_state.round_number >= max_rounds:
            st.balloons()
            st.header("🏆 Spiel beendet!")
        else:
            if st.button("Nächste Runde starten", type="primary"):
                self.save_state_for_undo()
                self.game_manager.proceed_to_next_round()
                st.rerun()

    # --- GUI PHASEN ---

    def setup_new_game(self):
        st.header("🧙‍♂️ Neues Wizard Spiel")
        col1, col2 = st.columns(2)
        with col1:
            num_players = st.selectbox("Anzahl Spieler", [3, 4, 5, 6], index=1, key="num_players_setup")
            round_number = st.selectbox("Startrunde", list(range(1, 21)), index=0, key="round_number_setup")

            # --- VALIDIERUNG ---
            max_rounds = 60 // num_players
            is_valid_round = round_number <= max_rounds

            if not is_valid_round:
                st.error(
                    f"Bei {num_players} Spielern können maximal {max_rounds} Runden gespielt werden (60 Karten). "
                    f"Bitte wählen Sie eine valide Startrunde."
                )
            
            st.subheader("Spieloptionen")
            deal_digitally = st.checkbox("Karten digital verteilen", value=True, key="deal_digitally")
            confirm_play = st.checkbox("Karten ausspielen bestätigen", value=False, key="confirm_play")
            
            # +++ NEUE CHECKBOX +++
            last_wizard_wins = st.checkbox("Letzter Zauberer gewinnt Stich", value=False, key="last_wizard_wins_rule", help="Wenn aktiv, gewinnt der letzte gespielte Zauberer den Stich. Standardmäßig gewinnt der erste.")
            # +++ ENDE NEUE CHECKBOX +++

        with col2:
            st.subheader("Spielernamen")
            for i in range(num_players):
                default_name = st.session_state.player_names.get(i, f"Spieler {i+1}")
                st.session_state.player_names[i] = st.text_input(
                    f"Name Spieler {i+1}",
                    value=default_name,
                    key=f"player_name_{i}"
                )

            st.subheader("Spielertypen")
            player_types = {i: st.selectbox(f"Typ {st.session_state.player_names[i]}", ["human", "computer"], key=f"player_type_{i}") for i in range(num_players)}
            human_players = [i for i, t in player_types.items() if t == "human"]
            human_player_id = st.selectbox("Hauptspieler (Sie)", human_players, format_func=lambda x: self.get_player_name(x)) if human_players else 0
            dealer_id = st.selectbox(
                "Kartengeber (Runde 1)",
                list(range(num_players)),
                format_func=lambda x: self.get_player_name(x),
                key="dealer_select"
            )

        if st.button("Spiel erstellen", disabled=not is_valid_round):
            self.save_state_for_undo()
            # +++ AUFRUF AKTUALISIERT +++
            self.game_manager.start_new_game(
                num_players, 
                player_types, 
                human_player_id, 
                round_number, 
                deal_digitally, 
                confirm_play,
                last_wizard_wins,
                dealer_id
            )
            # +++ ENDE AKTUALISIERUNG +++
            st.rerun()            
    def human_hand_input_stage(self):
        st.header(f"📜 Runde {self.game_manager.game_state.round_number} - Ihre Karten eingeben")
        st.info("Da die Option 'Karten digital verteilen' deaktiviert ist, geben Sie bitte Ihre Handkarten ein.")
        
        num_cards = self.game_manager.game_state.round_number
        player_id = self.game_manager.human_player_id
        
        st.subheader("Trumpfkarte wählen")
        trump_card = st.session_state.get("manual_trump_card")
        if trump_card is not None:
            st.image(self.create_card_image(trump_card, width=60, height=90), caption=str(trump_card))
        else:
            st.write("Keine Trumpfkarte gewählt")

        if "trump_suit_choice" not in st.session_state:
            suit_rows = [[Suit.RED, Suit.BLUE, Suit.GREEN], [Suit.YELLOW, Suit.WIZARD, Suit.JESTER]]
            for row in suit_rows:
                cols = st.columns(len(row))
                for idx, suit in enumerate(row):
                    if cols[idx].button(suit.value, key=f"trump_suit_{suit.value}"):
                        st.session_state.trump_suit_choice = suit.value
                        st.rerun()
            if st.button("Keine", key="trump_none"):
                st.session_state.manual_trump_card = None
                self.game_manager.set_trump_from_card(None)
                st.rerun()
        else:
            suit_val = st.session_state.trump_suit_choice
            if suit_val in [Suit.WIZARD.value, Suit.JESTER.value]:
                card = Card(Suit.WIZARD, 0) if suit_val == Suit.WIZARD.value else Card(Suit.JESTER, 0)
                if st.button("Trumpfkarte übernehmen", key=f"confirm_trump_{suit_val}"):
                    self.save_state_for_undo()
                    self.game_manager.set_trump_from_card(card)
                    st.session_state.manual_trump_card = card
                    del st.session_state.trump_suit_choice
                    st.rerun()
            else:
                st.write(f"Farbe **{suit_val}** gewählt - Wert auswählen:")
                value_rows = [range(1,6), range(6,11), range(11,14)]
                for row in value_rows:
                    cols = st.columns(len(row))
                    for idx, val in enumerate(row):
                        if cols[idx].button(str(val), key=f"trump_val_{val}"):
                            self.save_state_for_undo()
                            card = Card(Suit(suit_val), val)
                            self.game_manager.set_trump_from_card(card)
                            st.session_state.manual_trump_card = card
                            del st.session_state.trump_suit_choice
                            st.rerun()

        st.divider()

        entered_cards = self.game_manager.game_state.hands.get(player_id, [])
        st.subheader(f"Karten eingeben ({len(entered_cards)}/{num_cards})")

        if entered_cards:
            cols = st.columns(min(5, len(entered_cards)))
            for i, c in enumerate(entered_cards):
                with cols[i % 5]:
                    st.image(self.create_card_image(c, width=60, height=90), caption=str(c))

        if len(entered_cards) < num_cards:
            st.subheader(f"Karte {len(entered_cards)+1} auswählen")

            if 'manual_suit_choice' not in st.session_state:
                suit_rows = [[Suit.RED, Suit.BLUE, Suit.GREEN], [Suit.YELLOW, Suit.WIZARD, Suit.JESTER]]
                for row in suit_rows:
                    cols = st.columns(len(row))
                    for idx, suit in enumerate(row):
                        if cols[idx].button(suit.value, key=f"suit_btn_{len(entered_cards)}_{suit.value}"):
                            st.session_state.manual_suit_choice = suit.value
                            st.rerun()
            else:
                suit_val = st.session_state.manual_suit_choice
                if suit_val in [Suit.WIZARD.value, Suit.JESTER.value]:
                    card = Card(Suit.WIZARD, 1) if suit_val == Suit.WIZARD.value else Card(Suit.JESTER, 1)
                    if st.button("Karte übernehmen", key=f"confirm_special_{len(entered_cards)}"):
                        self.save_state_for_undo()
                        self.game_manager.game_state.hands[player_id].append(card)
                        del st.session_state.manual_suit_choice
                        st.rerun()
                else:
                    st.write(f"Farbe **{suit_val}** gewählt - Wert auswählen:")
                    value_rows = [range(1,6), range(6,11), range(11,14)]
                    for row in value_rows:
                        cols = st.columns(len(row))
                        for idx, val in enumerate(row):
                            if cols[idx].button(str(val), key=f"val_btn_{len(entered_cards)}_{val}"):
                                self.save_state_for_undo()
                                self.game_manager.game_state.hands[player_id].append(Card(Suit(suit_val), val))
                                del st.session_state.manual_suit_choice
                                st.rerun()
        else:
            if st.button("Karten & Trumpf bestätigen und zur Gebotsphase wechseln", type="primary"):
                try:
                    self.save_state_for_undo()
                    hand_cards = self.game_manager.game_state.hands.get(player_id, [])
                    self.game_manager.set_player_hand(player_id, hand_cards)
                    card = st.session_state.get("manual_trump_card")
                    self.game_manager.set_trump_from_card(card)
                    st.rerun()
                except Exception as e:
                    st.error(f"Fehler bei der Karteneingabe: {e}")

    def bidding_stage(self):
        game_state = self.game_manager.game_state
        st.header(f"📜 Runde {game_state.round_number} - Gebote abgeben")

        if self.game_manager.wizard_trump_chooser == self.game_manager.human_player_id:
            st.warning("Trumpfkarte war ein Zauberer. Bitte Trumpffarbe wählen:")
            choice = st.selectbox(
                "Trumpffarbe",
                [s.value for s in Suit if s not in [Suit.WIZARD, Suit.JESTER]],
                key="wizard_trump_select",
            )
            if st.button("Trumpf setzen", key="wizard_trump_confirm"):
                self.save_state_for_undo()
                self.game_manager.choose_trump_suit(choice)
                st.rerun()

        if self.game_manager.deal_digitally:
            st.subheader(f"Ihre Karten ({self.get_player_name(self.game_manager.human_player_id)})")
            main_col, trump_col = st.columns([4, 1])
            with trump_col:
                st.metric("Trumpf", game_state.trump_suit.value if game_state.trump_suit else "Keine")
            with main_col:
                hand = game_state.hands.get(self.game_manager.human_player_id, [])
                if hand:
                    cols = st.columns(len(hand))
                    for i, card in enumerate(hand):
                        with cols[i]:
                            st.image(self.create_card_image(card, width=60, height=90), caption=str(card))
        else:
             st.metric("Trumpf", game_state.trump_suit.value if game_state.trump_suit else "Keine")

        st.divider()
        st.subheader("Gebote der Spieler")
        
        if 'bids_in_progress' not in st.session_state:
            st.session_state.bids_in_progress = {p: -1 for p in game_state.players}

        for p in game_state.players:
            widget_key = f"bid_input_{p}"
            if widget_key in st.session_state:
                st.session_state.bids_in_progress[p] = st.session_state[widget_key]

        bid_cols = st.columns(len(game_state.players))
        for i in game_state.players:
            with bid_cols[i]:
                st.number_input(
                    f"{self.get_player_name(i)} ({'👤' if self.game_manager.player_types.get(i) == 'human' else '🤖'})",
                    min_value=-1, max_value=game_state.round_number, 
                    value=st.session_state.bids_in_progress[i], 
                    key=f"bid_input_{i}"
                )
                if st.button("Empfehlung", key=f"rec_btn_{i}"):
                    with st.spinner(f"Simuliere Gebote für {self.get_player_name(i)}..."):
                        st.session_state.bid_recommendations = self.bid_recommender.recommend_bid(game_state, i)
                        st.session_state.bid_rec_player = i
        
        computer_players_without_bid = [
            p for p in game_state.players 
            if self.game_manager.player_types.get(p) == 'computer' and st.session_state.bids_in_progress.get(p, -1) == -1
        ]
        if computer_players_without_bid:
            if st.button("🤖 Computer-Gebote ausfüllen", type="secondary"):
                self.fill_computer_bids(computer_players_without_bid)

        if 'bid_recommendations' in st.session_state:
            player, rec_df = st.session_state.bid_rec_player, st.session_state.bid_recommendations
            with st.expander(f"Gebot-Empfehlung für {self.get_player_name(player)}", expanded=True):
                if not rec_df.empty:
                    st.dataframe(
                        rec_df.style.format({'Mittelwert': '{:.2f}', 'Varianz': '{:.2f}', '95% Konfidenz Oben': '{:.2f}'}),
                        hide_index=True
                    )
                    best_bid = int(rec_df.iloc[0]['Gebot'])
                    st.success(f"Bestes Gebot: **{best_bid}** (Erwartete Punkte: {rec_df.iloc[0]['Mittelwert']:.2f})")
                else:
                    st.warning("Keine Empfehlung möglich.")

        if all(b != -1 for b in st.session_state.bids_in_progress.values()):
            if st.button("Spiel mit diesen Geboten starten", type="primary"):
                self.save_state_for_undo()
                self.game_manager.set_bids(st.session_state.bids_in_progress)
                del st.session_state.bids_in_progress
                if 'bid_recommendations' in st.session_state: del st.session_state.bid_recommendations
                st.rerun()
        else:
            st.warning("Bitte geben Sie alle Gebote ein, bevor das Spiel gestartet wird.")

    def play_game_stage(self):
        self.display_game_overview()
        self.display_current_trick()
        st.divider()
        self.card_input_interface()
        
        if self.game_manager.game_state.current_player == self.game_manager.human_player_id:
            self.make_ai_recommendations()

        st.divider()
        self.display_last_trick()

    # --- KARTENEINGABE & KI-METHODEN ---
    
    def fill_computer_bids(self, computer_players):
        game_state = self.game_manager.game_state
        with st.spinner("Computer berechnen ihre Gebote..."):
            for player_id in computer_players:
                if player_id not in game_state.hands or not game_state.hands[player_id]:
                    st.error(f"Fehler: Hand für Computerspieler {self.get_player_name(player_id)} nicht gefunden.")
                    continue
                recommendations = self.bid_recommender.recommend_bid(game_state, player_id)
                if not recommendations.empty:
                    best_bid = int(recommendations.iloc[0]['Gebot'])
                    st.session_state.bids_in_progress[player_id] = best_bid
        st.rerun()

    def card_input_interface(self):
        game_state = self.game_manager.game_state
        current_player = game_state.current_player
        player_type = self.game_manager.player_types.get(current_player, "human")
        
        st.subheader(f"🎯 {self.get_player_name(current_player)} ist am Zug")

        if player_type == "computer":
            if st.button(f"🤖 Computer Zug für {self.get_player_name(current_player)}", type="primary"):
                self.save_state_for_undo()
                self.play_computer_move(current_player)
        elif current_player == self.game_manager.human_player_id:
            self.main_player_card_input(current_player)
        else: # Gilt für andere menschliche Spieler
            self.human_card_input(current_player)

    def main_player_card_input(self, player_id: int):
        game_state = self.game_manager.game_state
        hand = sorted(game_state.hands.get(player_id, []))
        st.subheader(f"🃏 Deine Karten ({self.get_player_name(player_id)})")
        
        # Die Logik für das Bestätigen des Spielens bleibt erhalten
        confirm_logic_active = self.game_manager.confirm_play and 'selected_card_to_play' in st.session_state
        
        if confirm_logic_active:
            card_to_play = st.session_state.selected_card_to_play
            st.warning(f"Möchtest du wirklich **{card_to_play}** spielen?")
            c1, c2 = st.columns(2)
            if c1.button("Ja, spielen!", use_container_width=True, type="primary"):
                self.save_state_for_undo()
                self.game_manager.play_card(player_id, card_to_play)
                del st.session_state.selected_card_to_play
                if 'last_recommendations' in st.session_state:
                    del st.session_state.last_recommendations
                st.rerun()
            if c2.button("Nein, abbrechen", use_container_width=True):
                del st.session_state.selected_card_to_play
                st.rerun()

        cards_per_row = 10
        for i in range(0, len(hand), cards_per_row):
            row_cards = hand[i:i+cards_per_row]
            cols = st.columns(len(row_cards))
            for j, card in enumerate(row_cards):
                with cols[j]:
                    # Die Validierung hier stellt sicher, dass man nur Karten spielen kann, die den Regeln entsprechen
                    is_valid = WizardRules.is_valid_play(card, hand, game_state.current_trick_cards, game_state.trump_suit)
                    st.image(self.create_card_image(card, width=80, height=120))
                    if st.button("Spielen", key=f"play_{card}", use_container_width=True, disabled=not is_valid):
                        if self.game_manager.confirm_play:
                            st.session_state.selected_card_to_play = card
                        else:
                            self.save_state_for_undo()
                            self.game_manager.play_card(player_id, card)
                            if 'last_recommendations' in st.session_state:
                                del st.session_state.last_recommendations
                        st.rerun()
    
    # NEU: Überarbeitete Methode für die Eingabe anderer menschlicher Spieler
    def human_card_input(self, player_id: int):
        st.info(f"Bitte gib die gespielte Karte von {self.get_player_name(player_id)} ein.")
        game_state = self.game_manager.game_state

        suit_key = f"suit_choice_{player_id}"

        if suit_key not in st.session_state:
            suit_rows = [[Suit.RED, Suit.BLUE, Suit.GREEN], [Suit.YELLOW, Suit.WIZARD, Suit.JESTER]]
            for row in suit_rows:
                cols = st.columns(len(row))
                for idx, suit in enumerate(row):
                    if cols[idx].button(suit.value, key=f"h_suit_{player_id}_{suit.value}"):
                        st.session_state[suit_key] = suit.value
                        st.rerun()
        else:
            suit_val = st.session_state[suit_key]
            if suit_val in [Suit.WIZARD.value, Suit.JESTER.value]:
                card = Card(Suit.WIZARD, 1) if suit_val == Suit.WIZARD.value else Card(Suit.JESTER, 1)
                if st.button("Karte spielen", key=f"play_special_{player_id}"):
                    self.save_state_for_undo()
                    self.game_manager.play_card(player_id, card)
                    del st.session_state[suit_key]
                    st.rerun()
            else:
                st.write(f"Farbe **{suit_val}** gewählt - Wert auswählen:")
                value_rows = [range(1,6), range(6,11), range(11,14)]
                for row in value_rows:
                    cols = st.columns(len(row))
                    for idx, val in enumerate(row):
                        if cols[idx].button(str(val), key=f"h_val_{player_id}_{val}"):
                            self.save_state_for_undo()
                            self.game_manager.play_card(player_id, Card(Suit(suit_val), val))
                            del st.session_state[suit_key]
                            st.rerun()


    def play_computer_move(self, player_id: int):
        game_state = self.game_manager.game_state
        try:
            with st.spinner("Computer denkt nach..."):
                recommendations = self.assistant.get_recommendation(game_state, player_id)
            if recommendations:
                best_card = max(recommendations.items(), key=lambda x: x[1])[0]
                self.game_manager.play_card(player_id, best_card)
                st.success(f"🤖 Computer ({self.get_player_name(player_id)}) spielt: {best_card}")
                st.rerun()
            else:
                st.error(f"Computer konnte keine gültige Karte finden.")
        except Exception as e:
            st.error(f"Fehler beim Computer-Zug: {str(e)}")
            
    def make_ai_recommendations(self):
        st.subheader("KI Unterstützung")
        if st.button("🤖 KI-Empfehlung anfordern", type="secondary"):
            game_state = self.game_manager.game_state
            player_id = self.game_manager.human_player_id
            with st.spinner("KI analysiert Spielsituation..."):
                recommendations = self.assistant.get_recommendation(game_state, player_id)
            if recommendations:
                sorted_recs = sorted(recommendations.items(), key=lambda kv: kv[1], reverse=True)
                rec_table = [{"Karte": str(card), "Score": f"{score:.3f}"} for card, score in sorted_recs]
                st.session_state.last_recommendations = rec_table
            else:
                st.session_state.last_recommendations = None
        
        if "last_recommendations" in st.session_state and st.session_state.last_recommendations:
            with st.expander("Letzte KI-Empfehlung", expanded=True):
                st.table(st.session_state.last_recommendations)

    # --- STEUERUNG & HAUPTFUNKTION ---

    def _reset_for_new_game(self):
        """Clears gameplay related session state but keeps setup options."""
        keep_keys = {
            "player_names",
            "num_players_setup",
            "round_number_setup",
            "deal_digitally",
            "confirm_play",
            "last_wizard_wins_rule",
            "dealer_select",
            "ai_card_sims",
            "ai_card_dists",
            "ai_bid_sims",
            "sb_card_sims",
            "sb_card_dists",
            "sb_bid_sims",
        }
        keep_prefixes = ("player_name_", "player_type_")

        for key in list(st.session_state.keys()):
            if key in keep_keys or any(key.startswith(p) for p in keep_prefixes):
                continue
            del st.session_state[key]

        st.session_state.show_new_game_confirm = False
        st.rerun()

    def sidebar_controls(self):
        with st.sidebar:
            st.header("🎮 Spielkontrolle")

            if 'show_new_game_confirm' not in st.session_state:
                st.session_state.show_new_game_confirm = False

            if st.button("🔄 Neues Spiel starten"):
                st.session_state.show_new_game_confirm = True

            if st.session_state.show_new_game_confirm:
                st.warning("Sind Sie sicher, dass Sie ein neues Spiel starten wollen? Der aktuelle Fortschritt geht verloren.")
                c1, c2 = st.columns(2)
                if c1.button("Ja", key="confirm_new_game"):
                    self._reset_for_new_game()
                if c2.button("Nein", key="cancel_new_game"):
                    st.session_state.show_new_game_confirm = False

            if self.game_manager.game_phase != "setup":
                 if st.button("↩️ Letzte Aktion rückgängig machen"):
                      self.undo_last_action()
            
            if self.game_manager.game_state:
                st.divider()
                st.header("🤖 KI-Einstellungen")
                st.subheader("Karten-Empfehlung")
                card_sims = st.slider("Simulationen gesamt", 100, 5000, st.session_state.ai_card_sims, 100, key="sb_card_sims")
                card_dists = st.slider("Anzahl Verteilungen", 10, 500, st.session_state.ai_card_dists, 10, key="sb_card_dists")
                st.subheader("Gebots-Empfehlung")
                bid_sims = st.slider("Anzahl Simulationen", 100, 2000, st.session_state.ai_bid_sims, 100, key="sb_bid_sims")
                
                if st.button("KI-Parameter anwenden"):
                    st.session_state.ai_card_sims = card_sims
                    st.session_state.ai_card_dists = card_dists
                    st.session_state.ai_bid_sims = bid_sims
                    st.success("KI-Parameter aktualisiert!")
                    st.rerun()

    def run(self):
        st.title("🧙‍♂️ Wizard Assistant GUI")
        #img = Image.open("./1F7B18D8-2B4B-4FB2-BE97-EE999F377E35.png")
        col1, col2 = st.columns([1, 1])
        with col1:
           st.image("1F7B18D8-2B4B-4FB2-BE97-EE999F377E35.png",use_container_width=True)
        self.sidebar_controls()
        
        game_phase = self.game_manager.game_phase

        if game_phase == "setup":
            self.setup_new_game()
        elif game_phase == "bidding" and not self.game_manager.deal_digitally and not self.game_manager.is_human_hand_set():
            self.human_hand_input_stage()
        elif game_phase == "bidding":
            self.bidding_stage()
        elif game_phase == "play_game":
            self.play_game_stage()
        elif game_phase == "round_over":
            self.display_round_results()
            st.divider()
            self.display_last_trick()

def main():
    gui = WizardGUI()
    gui.run()

if __name__ == "__main__":
    main()
