# wizard_gui.py

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import copy
from collections import Counter

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

        # Initialisiere KI-Parameter und Spiel-Einstellungen im Session State, falls nicht vorhanden
        self.init_ai_params()
        self.init_game_settings()
        
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
            st.session_state.ai_card_sims = 2000
        if 'ai_card_dists' not in st.session_state:
            st.session_state.ai_card_dists = 200
        if 'ai_bid_sims' not in st.session_state:
            st.session_state.ai_bid_sims = 800

    def init_game_settings(self):
        """Initialisiert die Spiel-Einstellungen im Session State, um sie über Spiele hinweg zu speichern."""
        if 'game_settings' not in st.session_state:
            st.session_state.game_settings = {
                "num_players": 4,
                "start_round": 1,
                "deal_digitally": True,
                "confirm_play": False,
                "last_wizard_wins_rule": False,
                "dealer_id": 0,
                "player_types": {i: "human" if i == 0 else "computer" for i in range(4)},
                "human_player_id": 0
            }

    # --- HILFSMETHODEN (inkl. Undo-Logik) ---

    def save_state_for_undo(self):
        st.session_state.history.append(copy.deepcopy(st.session_state.game_manager))

    def undo_last_action(self):
        if st.session_state.history:
            st.session_state.game_manager = st.session_state.history.pop()
            # Bereinige temporäre Zustände, um die GUI zu aktualisieren
            for key in ['selected_card_to_play', 'bid_recommendations', 'last_recommendations', 'bids_in_progress', 'trump_suit_choice', 'manual_suit_choice', 'manual_trump_suit_choice']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        else:
            st.toast("Keine weiteren Aktionen zum Rückgängigmachen vorhanden.")
            
    def _handle_trump_suit_change(self):
        """Callback-Funktion, die bei Änderung des Trumpf-Dropdowns ausgelöst wird."""
        if 'wizard_trump_selection_box' in st.session_state:
            new_suit_value = st.session_state.wizard_trump_selection_box
            self.save_state_for_undo()
            self.game_manager.choose_trump_suit(new_suit_value)

    # --- DARSTELLUNGS-METHODEN (unverändert) ---

    def create_card_image(self, card: Card, width: int = 80, height: int = 120) -> Image.Image:
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width-1, height-1], outline='black', width=2)
        color_map = {Suit.RED: 'red', Suit.BLUE: 'blue', Suit.GREEN: 'green', Suit.YELLOW: 'orange', Suit.WIZARD: 'purple', Suit.JESTER: 'black'}
        color = color_map.get(card.suit, 'black')
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
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
        
        trump_display = game_state.trump_suit.value if game_state.trump_suit else "Keine"
        st.metric("Trumpf", trump_display)

        overview_data = [{
                "Spieler": f"{self.get_player_name(p)} ({'👤' if self.game_manager.player_types.get(p) == 'human' else '🤖'})",
                "Gebot": game_state.bids.get(p, 0), "Stiche": game_state.tricks_won.get(p, 0),
                "Punkte": self.game_manager.current_round_scores.get(p, 0),
                "Gesamt": self.game_manager.total_scores.get(p, 0)
            } for p in game_state.players]
        st.dataframe(pd.DataFrame(overview_data), use_container_width=True, hide_index=True)


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
        settings = st.session_state.game_settings

        col1, col2 = st.columns(2)
        with col1:
            num_players = st.selectbox("Anzahl Spieler", [3, 4, 5, 6], index=[3, 4, 5, 6].index(settings["num_players"]), key="num_players_setup")
            max_rounds = 60 // num_players
            
            valid_start_round = min(settings["start_round"], max_rounds)
            round_number = st.selectbox("Startrunde", list(range(1, max_rounds + 1)), index=valid_start_round - 1, key="round_number_setup")

            st.subheader("Spieloptionen")
            deal_digitally = st.checkbox("Karten digital verteilen", value=settings["deal_digitally"], key="deal_digitally_setup")
            confirm_play = st.checkbox("Karten ausspielen bestätigen", value=settings["confirm_play"], key="confirm_play_setup")
            last_wizard_wins = st.checkbox("Letzter Zauberer gewinnt Stich", value=settings["last_wizard_wins_rule"], key="last_wizard_wins_setup", help="Wenn aktiv, gewinnt der letzte gespielte Zauberer den Stich. Standardmäßig gewinnt der erste.")

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
            current_player_types = settings.get("player_types", {})
            player_types = {i: st.selectbox(f"Typ {self.get_player_name(i)}", ["human", "computer"], index=["human", "computer"].index(current_player_types.get(i, "computer")), key=f"player_type_{i}") for i in range(num_players)}
            
            human_players = [i for i, t in player_types.items() if t == "human"]
            
            main_player_default_index = 0
            if settings["human_player_id"] in human_players:
                main_player_default_index = human_players.index(settings["human_player_id"])
            human_player_id = st.selectbox("Hauptspieler (Sie)", human_players, index=main_player_default_index, format_func=lambda x: self.get_player_name(x)) if human_players else 0

            dealer_default_index = min(settings["dealer_id"], num_players - 1)
            dealer_id = st.selectbox(
                "Kartengeber (Runde 1)",
                list(range(num_players)),
                index=dealer_default_index,
                format_func=lambda x: self.get_player_name(x),
                key="dealer_select_setup"
            )

        if st.button("Spiel erstellen", type="primary"):
            self.save_state_for_undo()
            
            st.session_state.game_settings = {
                "num_players": num_players,
                "start_round": round_number,
                "deal_digitally": deal_digitally,
                "confirm_play": confirm_play,
                "last_wizard_wins_rule": last_wizard_wins,
                "dealer_id": dealer_id,
                "player_types": player_types,
                "human_player_id": human_player_id
            }

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
            st.rerun()            

    def human_hand_input_stage(self):
        st.header(f"📜 Runde {self.game_manager.game_state.round_number} - Karten eingeben")
        
        game_state = self.game_manager.game_state
        num_cards = game_state.round_number
        player_id = self.game_manager.human_player_id
        num_players = len(game_state.players)
        is_last_round = game_state.round_number == 60 // num_players

        # --- Teil 1: Trumpfkarte eingeben (nur wenn NICHT die letzte Runde) ---
        if not is_last_round:
            st.subheader("1. Trumpfkarte eingeben")
            trump_card = game_state.trump_card

            if trump_card:
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(self.create_card_image(trump_card, width=80, height=120), caption=f"Trumpf: {trump_card}")
                with col2:
                    if st.button("Trumpfkarte ändern", key="reset_trump_manual"):
                        self.save_state_for_undo()
                        self.game_manager.set_trump_from_card(None)
                        if 'manual_trump_suit_choice' in st.session_state:
                            del st.session_state.manual_trump_suit_choice
                        st.rerun()
            else:
                st.write("Wählen Sie die aufgedeckte Trumpfkarte aus:")
                if 'manual_trump_suit_choice' not in st.session_state:
                    suit_rows = [[Suit.RED, Suit.BLUE, Suit.GREEN], [Suit.YELLOW, Suit.WIZARD, Suit.JESTER]]
                    for row in suit_rows:
                        cols = st.columns(len(row))
                        for idx, suit in enumerate(row):
                            if cols[idx].button(suit.value, key=f"trump_select_suit_{suit.value}"):
                                st.session_state.manual_trump_suit_choice = suit
                                st.rerun()
                else:
                    suit = st.session_state.manual_trump_suit_choice
                    if suit in [Suit.WIZARD, Suit.JESTER]:
                        self.save_state_for_undo()
                        card = Card(suit, 1 if suit == Suit.WIZARD else 0)
                        start_player = (game_state.dealer + 1) % len(game_state.players)
                        self.game_manager.set_trump_from_card(card, chooser=start_player)
                        del st.session_state.manual_trump_suit_choice
                        st.rerun()
                    else:
                        st.write(f"Farbe **{suit.value}** gewählt - Wert auswählen:")
                        value_rows = [range(1, 6), range(6, 11), range(11, 14)]
                        for row in value_rows:
                            cols = st.columns(len(row))
                            for idx, val in enumerate(row):
                                if cols[idx].button(str(val), key=f"trump_select_val_{val}"):
                                    self.save_state_for_undo()
                                    card = Card(suit, val)
                                    start_player = (game_state.dealer + 1) % len(game_state.players)
                                    self.game_manager.set_trump_from_card(card, chooser=start_player)
                                    del st.session_state.manual_trump_suit_choice
                                    st.rerun()
            st.divider()

        # --- Teil 2: Handkarten eingeben ---
        st.subheader(f"Ihre Karten eingeben ({len(game_state.hands.get(player_id, []))}/{num_cards})")
        entered_cards = game_state.hands.get(player_id, [])

        if entered_cards:
            cols = st.columns(min(10, len(entered_cards) or 1))
            for i, c in enumerate(entered_cards):
                with cols[i % 10]:
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
                card_to_add = None
                if suit_val in [Suit.WIZARD.value, Suit.JESTER.value]:
                    card_to_add = Card(Suit(suit_val), 1)
                else:
                    st.write(f"Farbe **{suit_val}** gewählt - Wert auswählen:")
                    value_rows = [range(1,6), range(6,11), range(11,14)]
                    for row in value_rows:
                        cols = st.columns(len(row))
                        for idx, val in enumerate(row):
                            if cols[idx].button(str(val), key=f"val_btn_{len(entered_cards)}_{val}"):
                                card_to_add = Card(Suit(suit_val), val)
                
                if card_to_add:
                    self.save_state_for_undo()
                    self.game_manager.game_state.hands[player_id].append(card_to_add)
                    del st.session_state.manual_suit_choice
                    st.rerun()
        
        # --- Teil 3: Bestätigung ---
        if self.game_manager.is_human_hand_set():
            if st.button("Karten bestätigen und zur Gebotsphase wechseln", type="primary"):
                self.save_state_for_undo()
                st.rerun()
        else:
            if is_last_round:
                st.warning("Bitte alle Ihre Handkarten eingeben.")
            else:
                st.warning("Bitte zuerst die Trumpfkarte und alle Ihre Handkarten eingeben.")


    def bidding_stage(self):
        game_state = self.game_manager.game_state
        st.header(f"📜 Runde {game_state.round_number} - Gebote abgeben")

        # --- ZENTRALES TRUMPF-WIDGET ---
        st.subheader("Trumpf")
        trump_card = game_state.trump_card
        col1, col2 = st.columns([1, 4])

        with col1:
            if trump_card:
                st.image(self.create_card_image(trump_card, width=100, height=150), caption=f"Aufgedeckt: {trump_card}")
            else:
                st.info("Keine Trumpfkarte (letzte Runde).")
        
        with col2:
            is_choice_pending = self.game_manager.wizard_trump_chooser is not None and game_state.trump_suit is None
            
            if is_choice_pending:
                chooser_id = self.game_manager.wizard_trump_chooser
                chooser_type = self.game_manager.player_types.get(chooser_id, "human")

                if chooser_type == 'computer':
                    st.warning(f"Trumpfkarte ist ein Zauberer! **{self.get_player_name(chooser_id)} (🤖)** wählt die Farbe.")
                    if st.button(f"🤖 {self.get_player_name(chooser_id)} wählen lassen"):
                        self.save_state_for_undo()
                        hand = game_state.hands.get(chooser_id, [])
                        suit_counts = Counter(c.suit for c in hand if c.suit in [Suit.RED, Suit.BLUE, Suit.GREEN, Suit.YELLOW])
                        if suit_counts:
                            most_common_suit = suit_counts.most_common(1)[0][0]
                            self.game_manager.choose_trump_suit(most_common_suit.value)
                        else: # Fallback
                            self.game_manager.choose_trump_suit(Suit.RED.value)
                        st.rerun()
                else: # Human chooser
                    st.warning(f"Trumpfkarte ist ein Zauberer! **{self.get_player_name(chooser_id)} (👤)** muss die Farbe wählen.")
                    is_this_human_the_chooser = self.game_manager.human_player_id == chooser_id
                    options = [s.value for s in Suit if s not in [Suit.WIZARD, Suit.JESTER]]
                    
                    st.selectbox(
                        "Wähle die Trumpffarbe:",
                        options,
                        key="wizard_trump_selection_box",
                        disabled=not is_this_human_the_chooser,
                        on_change=self._handle_trump_suit_change
                    )
            else:
                final_trump_suit = game_state.trump_suit.value if game_state.trump_suit else "Keine"
                st.text_input("Trumpffarbe", value=final_trump_suit, disabled=True, key="trump_display")

        # --- ANZEIGE DER KARTEN & GEBOTE ---
        if self.game_manager.deal_digitally:
            st.subheader(f"Ihre Karten ({self.get_player_name(self.game_manager.human_player_id)})")
            hand = game_state.hands.get(self.game_manager.human_player_id, [])
            if hand:
                cols = st.columns(len(hand) or 1)
                for i, card in enumerate(hand):
                    with cols[i]:
                        st.image(self.create_card_image(card, width=60, height=90), caption=str(card))

        st.divider()
        st.subheader("Gebote der Spieler")
        
        if 'bids_in_progress' not in st.session_state:
            st.session_state.bids_in_progress = {p: -1 for p in game_state.players}

        # Blockiere Gebote, bis Trumpf feststeht
        bidding_disabled = is_choice_pending
        if bidding_disabled:
            st.info("Bitte warten, bis die Trumpffarbe gewählt wurde.")

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
                    key=f"bid_input_{i}",
                    disabled=bidding_disabled
                )
                if st.button("Empfehlung", key=f"rec_btn_{i}", disabled=bidding_disabled):
                    with st.spinner(f"Simuliere Gebote für {self.get_player_name(i)}..."):
                        st.session_state.bid_recommendations = self.bid_recommender.recommend_bid(game_state, i)
                        st.session_state.bid_rec_player = i
        
        computer_players_without_bid = [
            p for p in game_state.players 
            if self.game_manager.player_types.get(p) == 'computer' and st.session_state.bids_in_progress.get(p, -1) == -1
        ]
        if computer_players_without_bid:
            if st.button("🤖 Computer-Gebote ausfüllen", type="secondary", disabled=bidding_disabled):
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

        if all(b != -1 for b in st.session_state.bids_in_progress.values()) and not bidding_disabled:
            if st.button("Spiel mit diesen Geboten starten", type="primary"):
                self.save_state_for_undo()
                self.game_manager.set_bids(st.session_state.bids_in_progress)
                del st.session_state.bids_in_progress
                if 'bid_recommendations' in st.session_state: del st.session_state.bid_recommendations
                st.rerun()
        elif not bidding_disabled:
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

    # --- KARTENEINGABE & KI-METHODEN (unverändert, ausser fill_computer_bids) ---
    
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
            cols = st.columns(len(row_cards) or 1)
            for j, card in enumerate(row_cards):
                with cols[j]:
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
            card_to_play = None
            if suit_val in [Suit.WIZARD.value, Suit.JESTER.value]:
                card_to_play = Card(Suit(suit_val), 1)
            else:
                st.write(f"Farbe **{suit_val}** gewählt - Wert auswählen:")
                value_rows = [range(1,6), range(6,11), range(11,14)]
                for row in value_rows:
                    cols = st.columns(len(row))
                    for idx, val in enumerate(row):
                        if cols[idx].button(str(val), key=f"h_val_{player_id}_{val}"):
                            card_to_play = Card(Suit(suit_val), val)

            if card_to_play:
                self.save_state_for_undo()
                self.game_manager.play_card(player_id, card_to_play)
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
        """Clears gameplay related session state but keeps setup options and player names."""
        keys_to_keep = {
            "player_names",
            "game_settings",
            # Behalte auch die KI-Slider-Werte
            "ai_card_sims",
            "ai_card_dists",
            "ai_bid_sims",
            "sb_card_sims",
            "sb_card_dists",
            "sb_bid_sims",
        }
        
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                # Behalte auch die persistenten Widget-Keys für die Einstellungen
                if not key.endswith(('_setup', '_select_setup')):
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
                    st.rerun()

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
                bid_sims = st.slider("Anzahl Simulationen", 100, 5000, st.session_state.ai_bid_sims, 100, key="sb_bid_sims")
                
                if st.button("KI-Parameter anwenden"):
                    st.session_state.ai_card_sims = card_sims
                    st.session_state.ai_card_dists = card_dists
                    st.session_state.ai_bid_sims = bid_sims
                    st.success("KI-Parameter aktualisiert!")
                    # Re-initialize AI components with new values
                    self.assistant = BayesOptimalWizardAssistant(num_simulations=card_sims, num_distributions=card_dists)
                    self.bid_recommender = BayesOptimalBidRecommender(num_simulations=bid_sims)
                    st.rerun()

    def run(self):
        st.title("🧙‍♂️ Wizard Assistant GUI")
        st.write("Willkommen! Dieser Assistent soll dir beim Kartenspiel Wizard helfen. Du kannst entweder mit 'Karten digital verteilen' gegen den Computer trainieren oder mit physischen mit anderen Mitspielern spielen. Die Oberfläche kann noch einen Feinschliff gebauchen, trage gerne etwas bei, wenn du möchtest!")  
        try:
            st.image("1F7B18D8-2B4B-4FB2-BE97-EE999F377E35.png", use_container_width=True)
        except Exception:
            st.info("Titelbild nicht gefunden.")

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
