from wizard_assistant_o3 import *
from bid_recommender import BayesOptimalBidRecommender


class WizardAssistantValidator:
    def __init__(self):
        self.game_sim = WizardGameSimulator()
        # Für Tests weniger aufwändige Parameter
        self.assistant = BayesOptimalWizardAssistant(num_simulations=100, num_distributions=10)
    
    def test_basic_functionality(self) -> bool:
        print("Testing basic functionality...")
        try:
            game_state = self.game_sim.create_test_game_state(round_number=5, current_trick_idx=2, num_cards_in_current_trick=1)
            player_to_test = game_state.current_player
            
            recommendations = self.assistant.get_recommendation(game_state, player_to_test)
            print(f"✓ Generated recommendations for {len(recommendations)} cards (Player {player_to_test})")
            
            analysis = self.assistant.get_detailed_analysis(game_state, player_to_test)
            print(f"✓ Detailed analysis entries: {len(analysis['recommendations'])}")
            return True
        except Exception as e:
            print(f"✗ Basic functionality failed: {e}")
            import traceback; traceback.print_exc()
            return False
    
    def test_rule_compliance(self) -> bool:
        print("Testing rule compliance...")
        try:
            test_cases = [
                (3, 1, 0), (3, 1, 1),
                (8, 4, 2),
                (self.game_sim.max_rounds, self.game_sim.max_rounds - 1, self.game_sim.num_players - 1)
            ]
            for round_num, trick_idx, cards_in_trick in test_cases:
                if round_num > self.game_sim.max_rounds:
                    continue
                gs = self.game_sim.create_test_game_state(round_num, trick_idx, cards_in_trick)
                player_to_check = gs.current_player
                if not gs.hands.get(player_to_check):
                    continue
                recs = self.assistant.get_recommendation(gs, player_to_check)
                available = self.assistant.get_available_cards(gs, player_to_check)
                if not recs and available:
                    print(f"  WARNING: No recs for player {player_to_check}, though valid cards exist.")
                for rec_card, _ in recs.items():
                    valid = WizardRules.is_valid_play(
                        rec_card, gs.hands[player_to_check], gs.current_trick_cards, gs.trump_suit
                    )
                    if not valid:
                        print(f"✗ Invalid recommended card {rec_card} for Player {player_to_check}")
                        return False
            print("✓ All recommended cards are rule-compliant.")
            return True
        except Exception as e:
            print(f"✗ Rule compliance failed: {e}")
            import traceback; traceback.print_exc()
            return False
    
    def test_bayesian_learning(self) -> bool:
        print("Testing Bayesian learning (uncertainty reduction)...")
        try:
            gs = self.game_sim.create_test_game_state(round_number=6, current_trick_idx=2, num_cards_in_current_trick=1)
            player_to_test = gs.current_player
            if not self.assistant.get_available_cards(gs, player_to_test):
                print("✓ Skipped Bayesian learning test: No valid cards.")
                return True
            
            # 1) Initiale Empfehlung
            _ = self.assistant.get_recommendation(gs, player_to_test)
            initial = self.assistant.get_detailed_analysis(gs, player_to_test)
            
            # 2) Trainiere an (einfachen) Zuständen
            for i in range(5):
                ts = self.game_sim.create_test_game_state(round_number=6, current_trick_idx=i % 3, num_cards_in_current_trick=i % self.game_sim.num_players)
                tp = ts.current_player
                if self.assistant.get_available_cards(ts, tp):
                    _ = self.assistant.get_recommendation(ts, tp)
            
            # 3) Nach Training nochmals auf Ursprung
            _ = self.assistant.get_recommendation(gs, player_to_test)
            final = self.assistant.get_detailed_analysis(gs, player_to_test)
            
            if not initial['uncertainties']:
                print("✓ Inconclusive: Keine Unsicherheiten initial.")
                return True
            
            init_unc = np.mean(list(initial['uncertainties'].values()))
            fin_unc = np.mean(list(final['uncertainties'].values()))
            print(f"  Initial avg uncertainty: {init_unc:.2f}")
            print(f"  Final   avg uncertainty: {fin_unc:.2f}")
            if fin_unc < init_unc * 1.1:
                print("✓ Uncertainty decreased oder zumindest nicht deutlich gestiegen.")
            else:
                print("⚠ Uncertainty nicht gesunken oder deutlich gestiegen.")
            
            init_obs = sum(initial['observation_counts'].values())
            fin_obs = sum(final['observation_counts'].values())
            print(f"  Initial total observations: {init_obs}")
            print(f"  Final   total observations: {fin_obs}")
            if fin_obs > init_obs:
                print("✓ Observations counts increased.")
                return True
            else:
                print("✗ Beobachtungszähler stieg nicht.")
                return False
        except Exception as e:
            print(f"✗ Bayesian learning failed: {e}")
            import traceback; traceback.print_exc()
            return False
    
    def test_bidding_recommendations(self) -> bool:
        print("Testing bidding recommendations...")
        try:
            game_state = self.game_sim.create_test_game_state(round_number=5, current_trick_idx=0, num_cards_in_current_trick=0)
            player_to_test = game_state.current_player
            
            recommender = BayesOptimalBidRecommender(num_simulations=100)
            bid_recommendations = recommender.recommend_bid(game_state, player_to_test)
            
            if not bid_recommendations:
                print("✗ No bid recommendations generated.")
                return False
            
            print(f"✓ Generated bid recommendations: {bid_recommendations}")
            
            # Validate that all bids are within the valid range
            max_bid = len(game_state.hands.get(player_to_test, []))
            for bid in bid_recommendations.keys():
                if bid < 0 or bid > max_bid:
                    print(f"✗ Invalid bid {bid} for player {player_to_test}.")
                    return False
            
            print("✓ All bid recommendations are valid.")
            return True
        except Exception as e:
            print(f"✗ Bidding recommendations test failed: {e}")
            import traceback; traceback.print_exc()
            return False

    def run_all_tests(self) -> bool:
        print("="*50)
        print("WIZARD ASSISTANT VALIDATION")
        print("="*50)
        tests = [
            self.test_basic_functionality,
            self.test_rule_compliance,
            self.test_bayesian_learning,
            self.test_bidding_recommendations  # Add the new test here
        ]
        results = [t() for t in tests]
        passed = sum(results)
        print("\n" + "="*50)
        if passed == len(results):
            print(f"✓ ALL TESTS PASSED ({passed}/{len(results)})")
            return True
        else:
            print(f"✗ SOME TESTS FAILED ({passed}/{len(results)})")
            return False

###########
# Helferklasse zum Testen / Erzeugen von Zufalls-GameStates
###########

class WizardGameSimulator:
    """
    Erzeugt zufällige GameStates, um die Funktionalität zu prüfen.
    (Hier ist die Logik nur rudimentär, um z.B. mid‐round Zustände aufzusetzen.)
    """
    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.max_rounds = 60 // num_players  # Max. Kartenanzahl pro Spieler
    
    def create_test_game_state(
        self, round_number: int, current_trick_idx: int = 0, num_cards_in_current_trick: int = 0
    ) -> GameState:
        """
        Erzeugt einen zufälligen Spielzustand in „Runde round_number“, 
        mit bereits gespielten `current_trick_idx` Stichen, 
        und `num_cards_in_current_trick` Karten in diesem laufenden Stich.
        """
        deck = WizardDeck()
        deck.shuffle()
        players = list(range(self.num_players))
        hands: Dict[int, List[Card]] = {p: [] for p in players}
        
        # 1) Dealen der Karten für diese Runde
        for p in players:
            hands[p] = deck.deal_cards(round_number)
        
        # 2) Trumpf ziehen
        trump_card = None
        if deck.cards:
            trump_card = deck.deal_cards(1)[0]
        trump_suit = None
        if trump_card:
            if trump_card.suit not in [Suit.WIZARD, Suit.JESTER]:
                trump_suit = trump_card.suit
        
        # 3) Zufällige Bids
        bids = {p: random.randint(0, round_number) for p in players}
        tricks_won = {p: 0 for p in players}
        played_cards_globally: Set[Card] = set()
        
        current_trick_cards_on_table: List[Tuple[int, Card]] = []
        trick_leader = players[0]
        current_player_for_state = players[0]
        
        # 4) Vorherige Stiche (vereinfachte Logik): Wir spielen `current_trick_idx` Stiche 
        #    per Zufallskarte, um ein konsistentes "mid‐round" Setup zu bekommen.
        for i in range(current_trick_idx):
            temp_trick: List[Tuple[int, Card]] = []
            for p in players:
                if hands[p]:
                    c = hands[p].pop(0)
                    temp_trick.append((p, c))
                    played_cards_globally.add(c)
            if temp_trick:
                # Simplified: Der erste Spieler dieses Stichs gewinnt
                winner = temp_trick[0][0]
                tricks_won[winner] += 1
                trick_leader = winner
        
        current_player_for_state = trick_leader
        
        # 5) Karten in der aktuell laufenden Stich (num_cards_in_current_trick)
        for _ in range(num_cards_in_current_trick):
            if hands[current_player_for_state]:
                c = hands[current_player_for_state].pop(0)
                current_trick_cards_on_table.append((current_player_for_state, c))
                played_cards_globally.add(c)
                # nächster Spieler im Kreis
                idx = players.index(current_player_for_state)
                current_player_for_state = players[(idx + 1) % self.num_players]
            else:
                break
        
        return GameState(
            round_number=round_number,
            current_trick=current_trick_idx,
            trump_suit=trump_suit,
            players=players,
            hands=hands,
            bids=bids,
            tricks_won=tricks_won,
            current_trick_cards=current_trick_cards_on_table,
            played_cards=played_cards_globally,
            current_player=current_player_for_state,
            trick_leader=trick_leader
        )



def main():
    print("WIZARD ASSISTANT v5.0 – Multi-Perspective Learning mit Decision-Quality")
    print("="*50)
    validator = WizardAssistantValidator()
    all_passed = validator.run_all_tests()
    
    if all_passed:
        print("\nDemonstration mit Beispielzustand:")
        game_sim = WizardGameSimulator(num_players=4)
        example_state = game_sim.create_test_game_state(round_number=7, current_trick_idx=3, num_cards_in_current_trick=2)
        advise_player = example_state.current_player
        
        print(f"\n--- Example Game State ---")
        print(f"Round: {example_state.round_number}, Trump: {example_state.trump_suit}")
        print(f"Current Trick Index: {example_state.current_trick}")
        print(f"Cards in Current Trick: {example_state.current_trick_cards}")
        print(f"Trick Leader: {example_state.trick_leader}, Current Player: {advise_player}")
        print(f"Bids: {example_state.bids}")
        print(f"Tricks Won: {example_state.tricks_won}")
        print(f"Player {advise_player}'s Hand: {example_state.hands.get(advise_player, [])}")
        print(f"Played Cards Global: {len(example_state.played_cards)}")
        
        if not example_state.hands.get(advise_player):
            print("→ Player to advise has no cards. Keine Empfehlungen.")
        else:
            assistant = BayesOptimalWizardAssistant(num_simulations=2000, num_distributions=200)
            print(f"\n--- Getting Recommendation für Player {advise_player} ---")
            recs = assistant.get_recommendation(example_state, advise_player)
            if recs:
                print("\nEmpfohlene Karten (höher = besser):")
                sorted_recs = sorted(recs.items(), key=lambda kv: kv[1], reverse=True)
                for card, score in sorted_recs:
                    print(f"  Karte: {str(card):<15} Score: {score:.2f}")
                
                print("\n--- Detailed Analysis ---")
                analysis = assistant.get_detailed_analysis(example_state, advise_player)
                for card_str in analysis['recommendations']:
                    print(f"  Karte: {card_str}")
                    print(f"    Est. Reward: {analysis['card_evaluations'][card_str]:.2f}")
                    print(f"    Uncertainty: {analysis['uncertainties'][card_str]:.2f}")
                    lb, ub = analysis['confidence_intervals'][card_str]
                    print(f"    Confidence Interval: ({lb:.2f}, {ub:.2f})")
                    print(f"    Observations: {analysis['observation_counts'][card_str]}")
                    print(f"    UCB Score: {analysis['recommendations'][card_str]:.2f}")
            else:
                print("→ Keine Empfehlungen (keine valide Karte?).")

if __name__ == "__main__":
    main()