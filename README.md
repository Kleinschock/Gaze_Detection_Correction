# Gesture-Based Presentation Control

**Autor:** René Gökmen
## 1. Projektübersicht

Dieses Projekt implementiert ein robustes System zur Echtzeit-Steuerung von webbasierten Präsentationen (z.B. reveal.js) mittels Gesten, die über eine Standard-Webcam erfasst werden. Es verbindet moderne Computer-Vision-Techniken von MediaPipe zur 3D-Posenschätzung mit Deep-Learning-Modellen (GRU/FFNN), die in PyTorch und PyTorch Lightning entwickelt wurden, um eine intuitive und leistungsfähige Mensch-Computer-Interaktion zu ermöglichen.

Das System extrahiert 3D-Körper- und Hand-Keypoints aus dem Videostream, verarbeitet diese zu aussagekräftigen Merkmalen und klassifiziert sie mithilfe eines trainierten neuronalen Netzes. Erkannte Gesten wie "Wischen", "Rotieren" oder "Flippen" werden über WebSockets an eine Web-Präsentation gesendet und dort in Steuerungsbefehle umgesetzt.

### 2. Kernfunktionen

*   **Echtzeit-Gestenerkennung**: Nutzt ein GRU- oder FFNN-Modell zur Klassifizierung von Gesten aus einem Live-Kamerabild.
*   **3D-Posenschätzung**: Extrahiert 3D-Keypoints des Körpers mithilfe von MediaPipe für eine robuste, von der Kameraposition unabhängige Merkmals-Generierung.
*   **Aufmerksamkeitserkennung**: Analysiert die Kopfhaltung, um festzustellen, ob der Benutzer auf den Bildschirm blickt. Dies ermöglicht intelligente Features wie eine automatische Systemsperre bei Abwesenheit oder interaktive Tutorials bei Anwesenheit.
*   **Zustandsmanagement**: Implementiert ein Sperr-/Entsperrsystem (über die Geste `rolling`), um unbeabsichtigte Eingaben zu verhindern, sowie gestenspezifische Cooldowns zur Vermeidung von Mehrfacheingaben.
*   **Interaktive Slideshow-Demo**: Beinhaltet eine `reveal.js`-Präsentation, die über WebSockets mit dem Python-Backend kommuniziert und eine Live-Steuerung ermöglicht.
*   **Umfassende Konfiguration**: Nahezu alle Systemparameter, von der Modellarchitektur bis zu den Schwellenwerten für die Echtzeiterkennung, sind zentral in der Datei `src/config.py` konfigurierbar.
*   **Hyperparameter-Tuning**: Integriert mit `wandb` (Weights & Biases) für automatisierte Sweeps zur Optimierung der Hyperparameter.

## 3. Technische Pipeline

1.  **Kamera-Input**: `run_live.py` erfasst kontinuierlich Bilder von der Webcam mittels OpenCV.
2.  **Posenschätzung**: Jedes Bild wird an die MediaPipe Pose-Lösung weitergeleitet, die 3D-Landmarken des Körpers in Echtzeit extrahiert.
3.  **Merkmals-Präprozessierung**: Die rohen Landmarken-Koordinaten durchlaufen eine mehrstufige Verarbeitungspipeline (`preprocessing.py`):
    *   **Normalisierung**: Anwendung einer `body_centric`-Normalisierung für Invarianz gegenüber Position, Größe und Rotation.
    *   **Feature Engineering**: Berechnung dynamischer Merkmale wie Geschwindigkeit und Beschleunigung.
4.  **Modell-Inferenz**: Eine Sequenz der verarbeiteten Merkmalsvektoren wird an das geladene PyTorch-Lightning-Modell übergeben, das eine Wahrscheinlichkeitsverteilung über alle Gesten ausgibt.
5.  **Zustandsmanagement & Glättung**: Das System nutzt einen Puffer, um Vorhersagen über mehrere Frames zu glätten. Es verwaltet zudem den Systemzustand (gesperrt/entsperrt, Cooldowns), um eine zuverlässige Interaktion sicherzustellen.
6.  **WebSocket-Kommunikation**: Bei Erkennung einer validen Geste wird ein Befehl an den WebSocket-Server gesendet.
7.  **Slideshow-Steuerung**: Die `slideshow.html`-Seite empfängt die WebSocket-Nachrichten und nutzt die `Reveal.js`-API zur Steuerung der Präsentation.

### 3.1 Daten- und Merkmals-Pipeline

Die Qualität der Eingabedaten ist entscheidend für jedes Machine-Learning-Modell. Dieses Projekt verwendet eine wissenschaftlich fundierte, mehrstufige Feature-Engineering-Pipeline, um rohe 3D-Keypoint-Koordinaten in einen reichhaltigen, informativen Merkmalsdatensatz umzuwandeln. Dies wird sowohl für das Training (`data_loader.py`) als auch für die Live-Inferenz (`run_live.py`) identisch durchgeführt.

*   **Stufe 1: Konfigurierbare Keypoint-Auswahl**: Die genauen Keypoints, die für das Feature Engineering verwendet werden, sind in der `FEATURES_TO_USE`-Liste in `config.py` definiert. Die Skripte `data_loader.py` und `run_live.py` passen sich dynamisch an diese Liste an und stellen sicher, dass die gesamte Pipeline einen konsistenten Satz von Merkmalen verwendet.

*   **Stufe 2: Konfigurierbare Normalisierung/Standardisierung**: Das Projekt unterstützt zwei sich gegenseitig ausschließende Vorverarbeitungsstrategien, die über `PREPROCESSING_STRATEGY` in `config.py` gesteuert werden:
    1.  **`'body_centric'` (Standard)**: Um sicherzustellen, dass das Modell unabhängig von der Position, Größe und Ausrichtung des Benutzers ist, wenden wir an:
        *   **Positionsinvarianz**: Zentrierung aller Keypoints relativ zum Mittelpunkt der Schultern.
        *   **Größeninvarianz**: Skalierung aller Keypoints basierend auf der Schulterbreite des Benutzers.
        *   **Rotationsinvarianz**: Erstellung eines körperzentrierten Koordinatensystems.
    2.  **`'standardize'`**: Wendet die traditionelle Z-Score-Standardisierung an. Es berechnet den Mittelwert und die Standardabweichung für jedes Merkmal über den gesamten Trainingsdatensatz und skaliert die Daten entsprechend. Der angepasste Skalierer wird in `models/standard_scaler.json` gespeichert, um identische Transformationen während des Trainings, der Validierung und der Live-Inferenz zu gewährleisten.

*   **Stufe 3: Konfigurierbares Feature Engineering**: Nach der Normalisierung generiert die Pipeline zusätzliche Merkmale. Dieser Prozess ist vollständig modular und wird durch die `FEATURE_ENGINEERING_CONFIG` in `config.py` gesteuert.
    *   Die Logik für jedes Merkmal (z. B. **Geschwindigkeit**, **Beschleunigung**) ist in einer eigenen Funktion in `src/feature_engineering.py` isoliert.
    *   Die Hauptfunktion `preprocess_sequence` ruft diese Funktionen dynamisch auf der Grundlage der Konfiguration auf, was einfache Experimente durch einfaches Ein- und Ausschalten von Merkmalen ermöglicht.

Der endgültige Merkmalsvektor für jeden Frame kombiniert die normalisierten Keypoints mit allen zusätzlichen Merkmalen, die in der Konfiguration aktiviert sind, und schafft so eine reichhaltige, flexible und konsistente Eingabe für die neuronalen Netze.

## 4. Projektstruktur
```
├── Data/
│   └── train/           # Enthält Gesten-Daten im CSV-Format, nach Gesten geordnet.
├── models/              # Speichert trainierte Modell-Checkpoints (.ckpt).
├── results/             # Speichert Evaluationsergebnisse (Reports, Konfusionsmatrizen).
├── slideshow/
│   └── slideshow/       # Enthält die reveal.js-Präsentationsdateien (HTML, CSS, JS).
├── src/                 # Der gesamte Python-Quellcode des Projekts.
│   ├── class_balancing.py      # Stellt Strategien zum Ausgleich von unausgeglichenen Datensätzen bereit.
│   ├── config.py               # Zentrale Konfigurationsdatei für alle Parameter.
│   ├── data_loader.py          # Kern der Datenlade- und Vorbereitungspipeline für das Haupt-GRU-Modell.
│   ├── debug_data_loader.py    # Ein einfaches Hilfsskript zum Debuggen des Datenladeprozesses.
│   ├── evaluate.py             # Logik für die quantitative Evaluierung von trainierten Modellen.
│   ├── feature_engineering.py  # Enthält die gesamte Logik zur Berechnung einzelner Merkmale.
│   ├── finetune.py             # Logik für das Fine-Tuning eines Modells.
│   ├── head_pose_estimation.py # Modulare Komponente zur Erkennung der Kopfausrichtung des Benutzers.
│   ├── lightning_datamodule.py # Kapselt die PyTorch Lightning-Logik für das Laden von Daten.
│   ├── lightning_module.py     # Kapselt die PyTorch Lightning-Logik für Training, Validierung und Test.
│   ├── models.py               # Definiert die Architekturen der neuronalen Netze.
│   ├── preprocessing.py        # Zentrales Modul für alle Datenvorverarbeitungs- und Feature-Engineering-Aufgaben.
│   ├── run_live.py             # Logik für die Echtzeitanwendung.
│   ├── scaler.py               # Stellt eine StandardScaler-Klasse für die Z-Score-Normalisierung bereit.
│   ├── spotter_data_loader.py  # Dedizierter Datenlader für das binäre Spotter-Modell.
│   ├── test.py                 # Skript für schnelle Tests.
│   ├── train.py                # Logik für das Training der Hauptmodelle.
│   ├── train_spotter.py        # Logik für das Training des Spotter-Modells.
│   ├── tuning.py               # Skript zur Durchführung von Hyperparameter-Sweeps.
│   └── __init__.py             # Macht das src-Verzeichnis zu einem Python-Paket.
├── requirements.txt     # Python-Abhängigkeiten.
└── README.md            # Dieses Dokument.

```

## Setup & Installation

1. **Python 3.9+ installieren**
2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```
3. Stelle sicher, dass deine Daten im Ordner `Data/train/` liegen (wie vorgegeben).

## Training

Trainiere ein Modell (FFNN oder GRU):
```bash
python -m src.train --model gru   # oder --model ffnn
```
Die Modelle werden in `models/` gespeichert.

## Evaluation

Bewerte ein trainiertes Modell auf dem Test-Set:
```bash
python -m src.evaluate --model gru   # oder --model ffnn
```

Mit Hilfe von `--model_path` kann ein spezifischer Checkpoint ausgewählt werden.

For example:

```bash
python -m src.evaluate --model_path models/your_model_file.ckpt
```

Ergebnisse (z.B. Konfusionsmatrix) findest du in `results/`.

## Live-Demo (Gestensteuerung)

Starte die Echtzeit-Gestenerkennung und Präsentationssteuerung:
```bash
python -m src.run_live
```
- Die Anwendung simuliert Tastatur-Events (Pfeiltasten) für reveal.js.
- Feedback/Feedforward wird im Kamerabild angezeigt.

Zum Testen des besten Modells aus dem Bericht kann folgender Befehl verwendet werden: 
```bash
python -m src.run_live --model_path models/finetuned_gru_main_model-epoch=04-val_acc=0.9997.ckpt
```

## 5. Human-Computer Interaction (HCI) Konzepte

Ein nutzbares Echtzeitsystem erfordert mehr als nur ein genaues Modell. Dieses Projekt integriert mehrere HCI-Prinzipien:

*   **Sperr-/Entsperrmechanismus**: Das System startet in einem "gesperrten" Zustand, um versehentliche Aktivierungen zu verhindern. Eine einzige, dedizierte Geste (`rolling`, wie durch `TOGGLE_LOCK_GESTURE` in der Konfiguration definiert) wird verwendet, um zwischen dem "gesperrten" und "entsperrten" Zustand zu wechseln.
*   **Feedback**: Das System gibt dem Benutzer sofortiges visuelles Feedback. Die Benutzeroberfläche auf dem Bildschirm zeigt den aktuellen Zustand (Gesperrt/Aktiv), die erkannte Geste und einen visuellen Blitz zur Bestätigung, wenn eine Aktion ausgelöst wurde. Dies hält den Benutzer darüber informiert, was das System tut.
*   **Feedforward**: Die Benutzeroberfläche dient auch als Feedforward, indem sie den Benutzer passiv über den aktuellen Zustand des Systems informiert. Der Status "GESPERRT" zeigt dem Benutzer, dass er zuerst die Entsperrgeste ausführen muss.
*   **Cooldown und Pufferung**: Um zu verhindern, dass eine einzelne Geste wiederholt ausgelöst wird und um stabile Vorhersagen zu gewährleisten, verwendet das System nach jeder Aktion eine Abklingzeit und einen Vorhersagepuffer, um eine Geste über mehrere Frames zu bestätigen.

## Hinweise
- Die Datenstruktur in `Data/train/` bitte nicht verändern.
- Für eigene Gesten einfach neue Unterordner anlegen und CSVs ablegen.
---

**Autor:** René Gökmen
