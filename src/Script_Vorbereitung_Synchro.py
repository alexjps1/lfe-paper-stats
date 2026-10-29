from pathlib import Path
from datetime import datetime
import json
import re
import csv
import unicodedata


# ============================================================
# KONFIGURATION
# ============================================================

# Projektordner = Ordner, in dem dieses Script liegt
# (funktioniert so auf Mac und Windows)
PROJECT_ROOT = Path(__file__).resolve().parent


# Ordnernamen
DRIVING_FOLDER = "Fahrdaten_Eyetracking"
TUTOR_FOLDER = "WebApp_Onboarding"


# Ausgabeordner
OUTPUT_FOLDER = PROJECT_ROOT / "00_Analyse_Vorbereitung"
CONVERSATION_FOLDER = OUTPUT_FOLDER / "Gespraechsdaten"


# Probanden-ID: vier Buchstaben + mind. drei Zahlen
# ASC:  "Fahrt1_ADNA270.1.asc", "BEUA003.1.asc", "THKA0092.1.asc"
# JSON: "ADNA270.json"
PARTICIPANT_PATTERN = re.compile(
    r"(?:^|_)([A-Za-z]{4}\d{3,})(?:\.\d+)?$"
)


# Falls ASC- und JSON-Dateiname nicht übereinstimmen
# (Tippfehler): ASC-ID -> JSON-ID
ID_ALIASES = {
    "VAAN098": "VAAN089",
}


# Zeitfenster um jedes Gespräch (in Sekunden).
# Ein Gespräch läuft von "startedAt" bis zur letzten Nachricht ("at").
# Mit END_PADDING_S kann z.B. die Sprechdauer der letzten
# Antwort mit abgedeckt werden.
START_PADDING_S = 0.0
END_PADDING_S = 0.0


# Spalte in der ASC-Datei mit der Zeit in Millisekunden
# (0 = simulationStartedAt)
ASC_TIME_COLUMN = "Messzeitpunkt"


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def nfc(text):
    """
    Vereinheitlicht Umlaute (Mac speichert "ä" teils als "a" + "¨").
    """

    return unicodedata.normalize("NFC", text)


def find_participant_files(folder, extension):
    """
    Findet alle Dateien mit der gewünschten Endung
    und ordnet sie der Probanden-ID zu.
    """

    files = {}

    if not folder.exists():
        print(f"ORDNER NICHT GEFUNDEN: {folder}")
        return files

    for file in folder.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() != extension.lower():
            continue

        match = PARTICIPANT_PATTERN.search(file.stem)

        if not match:
            print(f"  Dateiname nicht erkannt: {file.name}")
            continue

        participant_id = match.group(1).upper()
        participant_id = ID_ALIASES.get(participant_id, participant_id)

        files[participant_id] = file

    return files


def find_groups():
    """
    Liefert alle Versuchsgruppen, die in beiden Ordnern vorkommen,
    als {Gruppenname: (ASC-Ordner, JSON-Ordner)}.
    """

    def subfolders(root):
        if not root.exists():
            print(f"ORDNER NICHT GEFUNDEN: {root}")
            return {}
        return {
            nfc(p.name): p
            for p in root.iterdir()
            if p.is_dir()
        }

    driving = subfolders(PROJECT_ROOT / DRIVING_FOLDER)
    tutor = subfolders(PROJECT_ROOT / TUTOR_FOLDER)

    return {
        name: (driving[name], tutor[name])
        for name in sorted(set(driving) & set(tutor))
    }


def load_json(file_path):
    """
    Lädt eine JSON-Datei.
    """

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(f"FEHLER beim Lesen von {file_path}: {e}")
        return None


def parse_time(value):
    """
    ISO-Zeitstempel ("2026-09-18T07:57:19.874Z") -> datetime
    """

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_conversation_windows(data):
    """
    Berechnet für jedes Gespräch das Zeitfenster in Millisekunden
    relativ zu simulationStartedAt (= Zeit 0 in der ASC-Datei).
    """

    sim_start = parse_time(data["simulationStartedAt"])

    def to_ms(value):
        return (parse_time(value) - sim_start).total_seconds() * 1000

    windows = []

    for index, conversation in enumerate(
        data.get("voiceConversations") or [],
        start=1
    ):

        messages = conversation.get("messages") or []

        window = {
            "conversation": index,
            "trigger": conversation.get("trigger"),
            "startedAt": conversation["startedAt"],
            "lastMessageAt": messages[-1]["at"] if messages else "",
            "n_messages": len(messages),
            "start_ms": to_ms(conversation["startedAt"])
                        - START_PADDING_S * 1000,
            "end_ms": None,
            "message_ms": [to_ms(m["at"]) for m in messages],
            "message_roles": [m.get("role", "") for m in messages],
            "n_rows": 0,
            "note": "",
        }

        if not messages:
            window["note"] = "keine Nachrichten"
        else:
            window["end_ms"] = (
                window["message_ms"][-1] + END_PADDING_S * 1000
            )

            if window["end_ms"] < 0:
                window["note"] = "vor Simulationsstart"
            elif window["start_ms"] < 0:
                window["note"] = "beginnt vor Simulationsstart"

        windows.append(window)

    return windows


def current_message(window, time_ms):
    """
    Index (1-basiert) und Rolle der letzten Nachricht,
    die zum Zeitpunkt time_ms schon gesendet war.
    """

    index = 0

    for i, message_ms in enumerate(window["message_ms"]):
        if message_ms <= time_ms:
            index = i + 1
        else:
            break

    if index == 0:
        return "", ""

    return index, window["message_roles"][index - 1]


def extract_conversation_rows(asc_file, windows, output_file, meta):
    """
    Liest die ASC-Datei zeilenweise und schreibt alle Rohdaten-Zeilen,
    die in ein Gesprächsfenster fallen, unverändert in output_file
    (mit zusätzlichen Spalten vorne).

    Gibt die letzte Zeit (ms) der ASC-Datei zurück.
    """

    active = [w for w in windows if w["end_ms"] is not None]

    extra_columns = [
        "group",
        "participant_id",
        "conversation",
        "trigger",
        "conversation_startedAt",
        "t_conversation_s",
        "message_index",
        "message_role",
    ]

    last_time_ms = None

    with open(
        asc_file, "r", encoding="latin-1", newline=""
    ) as src, open(
        output_file, "w", encoding="utf-8", newline=""
    ) as dst:

        header = src.readline().rstrip("\r\n")
        columns = header.split(",")

        if ASC_TIME_COLUMN not in columns:
            raise ValueError(
                f"Spalte '{ASC_TIME_COLUMN}' fehlt in {asc_file.name}"
            )

        time_index = columns.index(ASC_TIME_COLUMN)

        dst.write(",".join(extra_columns) + "," + header + "\n")

        for line in src:

            line = line.rstrip("\r\n")

            if not line:
                continue

            try:
                time_ms = float(line.split(",", time_index + 1)[time_index])
            except (ValueError, IndexError):
                continue

            last_time_ms = time_ms

            for window in active:

                if not (window["start_ms"] <= time_ms <= window["end_ms"]):
                    continue

                message_index, message_role = current_message(
                    window, time_ms
                )

                t_conversation = (
                    time_ms - window["start_ms"]
                ) / 1000 - START_PADDING_S

                prefix = [
                    meta["group"],
                    meta["participant_id"],
                    str(window["conversation"]),
                    window["trigger"] or "",
                    window["startedAt"],
                    f"{t_conversation:.3f}",
                    str(message_index),
                    message_role,
                ]

                dst.write(",".join(prefix) + "," + line + "\n")
                window["n_rows"] += 1

    return last_time_ms


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():

    print()
    print("=" * 70)
    print("KI-TUTOR / SILAB DATEN - SYNCHRONISIERUNG GESPRÄCHE <-> ASC")
    print("=" * 70)
    print(f"Projektordner: {PROJECT_ROOT}")
    print()

    CONVERSATION_FOLDER.mkdir(parents=True, exist_ok=True)

    overview_rows = []

    for group, (driving_path, tutor_path) in find_groups().items():

        print()
        print("-" * 70)
        print(f"VERSUCHSGRUPPE: {group}")
        print("-" * 70)

        asc_files = find_participant_files(driving_path, ".asc")
        json_files = find_participant_files(tutor_path, ".json")

        print(f"ASC-Dateien gefunden:  {len(asc_files)}")
        print(f"JSON-Dateien gefunden: {len(json_files)}")

        for pid in sorted(set(asc_files) - set(json_files)):
            print(f"  ASC ohne JSON: {pid}")

        for pid in sorted(set(json_files) - set(asc_files)):
            print(f"  JSON ohne ASC: {pid}")

        participant_ids = sorted(set(asc_files) & set(json_files))

        group_output = CONVERSATION_FOLDER / group
        group_output.mkdir(parents=True, exist_ok=True)

        for participant_id in participant_ids:

            json_file = json_files[participant_id]
            asc_file = asc_files[participant_id]

            data = load_json(json_file)

            if data is None:
                continue

            if not data.get("simulationStartedAt"):
                print(
                    f"  {participant_id}: kein simulationStartedAt "
                    f"(mode={data.get('mode')}) -> übersprungen"
                )
                continue

            windows = build_conversation_windows(data)

            if not windows:
                print(f"  {participant_id}: keine Gespräche")
                continue

            output_file = (
                group_output / f"{participant_id}_gespraeche.csv"
            )

            last_time_ms = extract_conversation_rows(
                asc_file,
                windows,
                output_file,
                {"group": group, "participant_id": participant_id},
            )

            # Plausibilitätscheck: ASC-Länge vs. Fahrtdauer laut JSON
            if data.get("driveEndedAt") and last_time_ms is not None:

                drive_s = (
                    parse_time(data["driveEndedAt"])
                    - parse_time(data["simulationStartedAt"])
                ).total_seconds()

                diff_s = last_time_ms / 1000 - drive_s

                if abs(diff_s) > 60:
                    print(
                        f"  WARNUNG {participant_id}: ASC-Länge weicht "
                        f"{diff_s:+.0f} s von der Fahrtdauer im JSON ab"
                    )

            total_rows = sum(w["n_rows"] for w in windows)

            print(
                f"  {participant_id}: {len(windows)} Gespräche, "
                f"{total_rows} Zeilen -> {output_file.name}"
            )

            for window in windows:

                note = window["note"]

                if (
                    window["end_ms"] is not None
                    and last_time_ms is not None
                    and window["start_ms"] > last_time_ms
                ):
                    note = "nach Ende der ASC-Daten"

                elif window["end_ms"] is not None and window["n_rows"] == 0 \
                        and not note:
                    note = "keine ASC-Zeilen im Fenster"

                overview_rows.append({
                    "group": group,
                    "participant_id": participant_id,
                    "json_file": json_file.name,
                    "asc_file": asc_file.name,
                    "simulationStartedAt": data["simulationStartedAt"],
                    "conversation": window["conversation"],
                    "trigger": window["trigger"],
                    "startedAt": window["startedAt"],
                    "lastMessageAt": window["lastMessageAt"],
                    "start_s": f"{window['start_ms'] / 1000:.3f}",
                    "end_s": (
                        f"{window['end_ms'] / 1000:.3f}"
                        if window["end_ms"] is not None else ""
                    ),
                    "n_messages": window["n_messages"],
                    "n_asc_rows": window["n_rows"],
                    "note": note,
                })

    # ========================================================
    # ÜBERSICHT SPEICHERN
    # ========================================================

    overview_file = OUTPUT_FOLDER / "gespraechs_uebersicht.csv"

    fieldnames = [
        "group",
        "participant_id",
        "json_file",
        "asc_file",
        "simulationStartedAt",
        "conversation",
        "trigger",
        "startedAt",
        "lastMessageAt",
        "start_s",
        "end_s",
        "n_messages",
        "n_asc_rows",
        "note",
    ]

    with open(
        overview_file, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overview_rows)

    print()
    print("=" * 70)
    print("FERTIG")
    print("=" * 70)
    print(f"Rohdaten pro Proband:  {CONVERSATION_FOLDER}")
    print(f"Übersicht Gespräche:   {overview_file}")
    print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
