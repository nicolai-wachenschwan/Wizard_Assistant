# wizard_gui.py

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import copy 
import os

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
    
    def display_game_overview(self):
        game_state = self.game_manager.game_state
        st.header(f"🎮 Runde {game_state.round_number} / Stich {game_state.current_trick + 1}")
        overview_data = [{
                "Spieler": f"Spieler {p+1} ({'👤' if self.game_manager.player_types.get(p) == 'human' else '🤖'})",
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
                    st.write(f"**Spieler {player+1}**")
                    st.image(self.create_card_image(card), caption=str(card))

    def display_last_trick(self):
        if self.game_manager.last_trick:
            st.subheader("🔄 Letzter Stich")
            last_trick = self.game_manager.last_trick
            trick_cols = st.columns(len(last_trick['cards']))
            for i, (player, card) in enumerate(last_trick['cards']):
                with trick_cols[i]:
                    is_winner = player == last_trick['winner']
                    st.write(f"**{'🏆 ' if is_winner else ''}Spieler {player+1}{' (Gewinner)' if is_winner else ''}**")
                    st.image(self.create_card_image(card), caption=str(card))

    def display_round_results(self):
        game_state = self.game_manager.game_state
        st.header(f"📊 Ergebnis von Runde {game_state.round_number}")
        results_data = [{
            "Spieler": f"Spieler {p+1}", "Gebot": game_state.bids.get(p, 0),
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
            st.subheader("Spielertypen")
            player_types = {i: st.selectbox(f"Spieler {i+1}", ["human", "computer"], key=f"player_type_{i}") for i in range(num_players)}
            human_players = [i for i, t in player_types.items() if t == "human"]
            human_player_id = st.selectbox("Hauptspieler (Sie)", human_players, format_func=lambda x: f"Spieler {x+1}") if human_players else 0

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
                last_wizard_wins # Neuer Parameter wird übergeben
            )
            # +++ ENDE AKTUALISIERUNG +++
            st.rerun()            
    def human_hand_input_stage(self):
        st.header(f"📜 Runde {self.game_manager.game_state.round_number} - Ihre Karten eingeben")
        st.info("Da die Option 'Karten digital verteilen' deaktiviert ist, geben Sie bitte Ihre Handkarten ein.")
        
        num_cards = self.game_manager.game_state.round_number
        player_id = self.game_manager.human_player_id
        
        with st.form(key="manual_hand_form"):
            st.subheader("Wählen Sie die Trumpffarbe")
            possible_trumps = [s.value for s in Suit if s not in [Suit.WIZARD, Suit.JESTER]]
            trump_suit_selection = st.selectbox(
                "Trumpffarbe für diese Runde",
                options=["Keine"] + possible_trumps, 
                key="trump_suit_manual"
            )
            st.divider()

            st.subheader(f"Geben Sie die {num_cards} Karten von Spieler {player_id + 1} ein")
            hand_input = []
            cols = st.columns(4)
            for i in range(num_cards):
                with cols[i % 4]:
                    st.write(f"Karte {i+1}")
                    suit = st.selectbox("Farbe", [s.value for s in Suit], key=f"suit_{i}")
                    
                    is_special_card = suit in [Suit.WIZARD.value, Suit.JESTER.value]
                    # Hier verwenden wir jetzt auch ein Selectbox für die Konsistenz
                    value_options = list(range(1, 14))
                    value = st.selectbox(
                        "Wert", 
                        options=value_options,
                        key=f"value_{i}",
                        disabled=is_special_card,
                        label_visibility="collapsed" if is_special_card else "visible"
                    )
                    hand_input.append({"suit": suit, "value": value})
            
            submitted = st.form_submit_button("Karten & Trumpf bestätigen und zur Gebotsphase wechseln")
            if submitted:
                try:
                    hand_cards = []
                    for card_data in hand_input:
                        if card_data["suit"] == Suit.WIZARD.value:
                            hand_cards.append(Card(Suit.WIZARD, 14))
                        elif card_data["suit"] == Suit.JESTER.value:
                            hand_cards.append(Card(Suit.JESTER, 0))
                        else:
                            hand_cards.append(Card(Suit(card_data["suit"]), card_data["value"]))
                    
                    # Validierung auf doppelte Karten, funktioniert nicht richtig, weil es 4 identische Wizards und Jester gibt
                    if False:#len(set(hand_cards)) != len(hand_cards):
                        st.error("Fehler: Sie haben eine oder mehrere Karten doppelt eingegeben. Bitte korrigieren Sie Ihre Eingabe.")
                    else:
                        self.save_state_for_undo() 
                        self.game_manager.set_player_hand(player_id, hand_cards)
                        selected_trump = st.session_state.trump_suit_manual
                        self.game_manager.set_trump_suit_manually(selected_trump if selected_trump != "Keine" else None)
                        st.rerun()

                except Exception as e:
                    st.error(f"Fehler bei der Karteneingabe: {e}")

    def bidding_stage(self):
        game_state = self.game_manager.game_state
        st.header(f"📜 Runde {game_state.round_number} - Gebote abgeben")

        if self.game_manager.deal_digitally:
            st.subheader(f"Ihre Karten (Spieler {self.game_manager.human_player_id + 1})")
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
                    f"Spieler {i+1} ({'👤' if self.game_manager.player_types.get(i) == 'human' else '🤖'})",
                    min_value=-1, max_value=game_state.round_number, 
                    value=st.session_state.bids_in_progress[i], 
                    key=f"bid_input_{i}"
                )
                if st.button("Empfehlung", key=f"rec_btn_{i}"):
                    with st.spinner(f"Simuliere Gebote für Spieler {i+1}..."):
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
            with st.expander(f"Gebot-Empfehlung für Spieler {player+1}", expanded=True):
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
                    st.error(f"Fehler: Hand für Computerspieler {player_id + 1} nicht gefunden.")
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
        
        st.subheader(f"🎯 Spieler {current_player + 1} ist am Zug")

        if player_type == "computer":
            if st.button(f"🤖 Computer Zug für Spieler {current_player + 1}", type="primary"):
                self.save_state_for_undo()
                self.play_computer_move(current_player)
        elif current_player == self.game_manager.human_player_id:
            self.main_player_card_input(current_player)
        else: # Gilt für andere menschliche Spieler
            self.human_card_input(current_player)

    def main_player_card_input(self, player_id: int):
        game_state = self.game_manager.game_state
        hand = sorted(game_state.hands.get(player_id, []))
        st.subheader(f"🃏 Deine Karten (Spieler {player_id+1})")
        
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
        st.info(f"Bitte gib die gespielte Karte von Spieler {player_id + 1} ein.")
        game_state = self.game_manager.game_state

        with st.form(key=f"human_input_form_{player_id}"):
            cols = st.columns(2)
            with cols[0]:
                suit_str = st.selectbox("Farbe", [s.value for s in Suit], key=f"suit_input_{player_id}")
            
            is_special = suit_str in [Suit.WIZARD.value, Suit.JESTER.value]
            
            with cols[1]:
                value = st.selectbox(
                    "Wert", 
                    options=list(range(1, 14)), 
                    key=f"value_input_{player_id}",
                    disabled=is_special
                )

            submitted = st.form_submit_button("Karte bestätigen")
            if submitted:
                # Karte aus Eingabe erstellen
                card = None
                if suit_str == Suit.WIZARD.value:
                    card = Card(Suit.WIZARD, 14)
                elif suit_str == Suit.JESTER.value:
                    card = Card(Suit.JESTER, 0)
                else:
                    card = Card(Suit(suit_str), value)
                
                # Validierung: Wurde die Karte bereits gespielt oder ist sie special und die Zahl der Special-Karten <4?
                wizard_count_valid= len([card for card in game_state.played_cards if card.suit == Suit.WIZARD])<=3
                jester_count_valid= len([card for card in game_state.played_cards if card.suit == Suit.JESTER])<=3
                #Validierung funktioniert noch nicht richtig, daher auskommentiert
                if False:#(card in game_state.played_cards and not is_special) or (card.suit == Suit.JESTER and not jester_count_valid) or (card.suit == Suit.WIZARD and not wizard_count_valid):
                    st.error(f"Fehler: Die Karte '{card}' wurde bereits gespielt und kann nicht erneut eingegeben werden.")
                else:
                    # Da wir die Hand des anderen Spielers nicht kennen, können wir is_valid_play nicht sinnvoll nutzen.
                    # Die Hauptvalidierung ist, dass die Karte nicht schon im Spiel ist.
                    self.save_state_for_undo()
                    self.game_manager.play_card(player_id, card)
                    st.rerun()


    def play_computer_move(self, player_id: int):
        game_state = self.game_manager.game_state
        try:
            with st.spinner("Computer denkt nach..."):
                recommendations = self.assistant.get_recommendation(game_state, player_id)
            if recommendations:
                best_card = max(recommendations.items(), key=lambda x: x[1])[0]
                self.game_manager.play_card(player_id, best_card)
                st.success(f"🤖 Computer (Spieler {player_id+1}) spielt: {best_card}")
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

    def sidebar_controls(self):
        with st.sidebar:
            st.header("🎮 Spielkontrolle")
            if st.button("🔄 Neues Spiel starten"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
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
                bid_sims = st.slider("Anzahl Simulationen", 100, 2000, st.session_state.ai_bid_sims, 100, key="sb_bid_sims")
                
                if st.button("KI-Parameter anwenden"):
                    st.session_state.ai_card_sims = card_sims
                    st.session_state.ai_card_dists = card_dists
                    st.session_state.ai_bid_sims = bid_sims
                    st.success("KI-Parameter aktualisiert!")
                    st.rerun()

    def run(self):
        st.title("🧙‍♂️ Wizard Assistant GUI")
        # --- title image anzeigen, falls vorhanden ---
        if os.path.exists("./1F7B18D8-2B4B-4FB2-BE97-EE999F377E35.png"):
            try:
                img = Image.open("./1F7B18D8-2B4B-4FB2-BE97-EE999F377E35.png")
                st.image(img, use_column_width=True)
            except Exception:
                # Falls die Datei nicht lesbar ist, einfach weiter ohne Fehler
                pass
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
