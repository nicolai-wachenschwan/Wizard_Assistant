# Wizard Assistant

Dieses Repository enthält den Code für einen simulationsbasierten Assistenten für das Kartenspiel "Wizard". Der Assistent ist in der Lage, sowohl Empfehlungen für das abzugebende Gebot (`Bid Recommendation`) als auch für die auszuspielende Karte (`Card Recommendation`) zu geben.

Der Kern des Assistenten basiert auf Monte-Carlo-Simulationen, die durch einen Bayes-optimalen Ansatz verfeinert werden, um aus den Simulationsergebnissen zu lernen und die Empfehlungen kontinuierlich zu verbessern.

## Inhaltsverzeichnis
- [Algorithmus zur Kartenempfehlung](#algorithmus-zur-kartenempfehlung)
  - [Flowchart: Kartenempfehlung](#flowchart-kartenempfehlung)
  - [Kerndetails des Algorithmus](#kerndetails-des-algorithmus)
- [Algorithmus zur Gebotsempfehlung](#algorithmus-zur-gebotsempfehlung)
  - [Flowchart: Gebotsempfehlung](#flowchart-gebotsempfehlung)
  - [Kerndetails des Algorithmus](#kerndetails-des-algorithmus-1)

---

## Algorithmus zur Kartenempfehlung

Die Kartenempfehlung ist der komplexeste Teil des Assistenten. Sie kombiniert eine große Anzahl von Simulationen mit einem Lernmechanismus, um die bestmögliche Karte in einer gegebenen Spielsituation zu finden.

### Flowchart: Kartenempfehlung
```mermaid
graph TD
    A["Start: Empfehlung für Spieler P anfordern"] --> B["Verfügbare Karten für P bestimmen"]
    B --> C["Unbekannte Karten identifizieren"]
    C --> D["Loop: N verschiedene Verteilungen der unbekannter Karten"]
    D --> E["Für jede verfügbare Karte C_i…"]
    E --> F["Erstelle neuen Spielzustand S', als ob P die Karte C_i gespielt hätte"]
    F --> G["Loop: M rein zufällige Simulationen"]
    G --> H["Simuliere Spiel von Zustand S' bis zum Ende der Runde"]
    H --> I["Finalen Reward R für Spieler P berechnen"]
    I --> J["Entscheidungspfad 'path' aufzeichnen"]
    J --> K["Bayes-optimale Backpropagation"]

    subgraph "Bayes-optimale Backpropagation"
        K --> K1["Reward R durch Pfad zurückpropagieren"]
        K1 --> K2["Reward für jeden Schritt zeitlich diskontieren (γ^t)"]
        K2 --> K3["Reward kausal gewichten (1.0 für Spieler P, 0.8 für andere)"]
        K3 --> K4["Reward-Beobachtung im BayesianRewardEstimator aktualisieren"]
        K4 --> K5["Update für spezifischen Schlüssel (Kontext + Karte)"]
        K5 --> K6["Update für allgemeinen Kontext-Schlüssel (Transfer Learning)"]
    end

    K6 --> G

    G -- "M-mal durchlaufen" --> L["Durchschnittlichen Reward für Karte C_i in dieser Verteilung berechnen"]
    L --> E
    E -- "Alle Karten durchlaufen" --> D
    D -- "N-mal durchlaufen" --> M["Finale Bewertung aller Karten"]

    subgraph "Finale Bewertung"
        M --> M1["Für jede Karte C_i…"]
        M1 --> M2["Hole geschätzten Reward aus BayesianRewardEstimator"]
        M2 --> M3["Hole Unsicherheit (Varianz) aus Estimator"]
        M3 --> M4["Berechne Upper Confidence Bound (UCB) Score"]
        M4 --> M
    end

    M --> N["Wähle Karte mit höchstem UCB-Score"]
    N --> O["Ende: Empfohlene Karte"]
