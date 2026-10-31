"""
FINALE EYETRACKING-AUSWERTUNG
=============================

Dieses Script berechnet alle Eyetracking-Kennzahlen direkt aus den
Rohdaten und schreibt sie in EINE Excel-Datei (Aufbau nach
"guide_datenauswertung"):

  Tabellenblatt          Inhalt
  ---------------------  ---------------------------------------------------
  Gesamtfahrt            1 Zeile pro Person (G1–G4), gesamte Fahrt
  Tutor_Abschnitte       1 Zeile pro Person (G3/G4) und Abschnitt:
                         Interaktion mit 60-s-Regel, Interaktion ohne Regel,
                         Fahrt außerhalb der Interaktionen
  Tutor_Interaktionen    1 Zeile pro Person, Interaktion (Block) und Variante
                         (inkl. Prüfung der 12-s-Regel)
  Variablen              Beschreibung und Einheit jeder Spalte
  Definitionen           AOIs, Zeiträume, Datenqualität, Parameter

Eingaben:
  Fahrdaten_Eyetracking/<Gruppe>/*.asc     Rohdaten Fahrsimulator + Eyetracking
  WebApp_Onboarding/<Gruppe>/*.json        simulationStartedAt (= Zeit 0 der ASC)
  InDrive_Interaktionsanalyse_...xlsx      Zeitfenster der Tutorinteraktionen
"""

from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# 1  KONFIGURATION
# ============================================================

# Projektordner = Ordner, in dem dieses Script liegt
PROJECT_ROOT = Path(__file__).resolve().parent

DRIVING_FOLDER = PROJECT_ROOT / "Fahrdaten_Eyetracking"
TUTOR_FOLDER = PROJECT_ROOT / "WebApp_Onboarding"
WINDOWS_FILE = PROJECT_ROOT / \
    "InDrive_Interaktionsanalyse_v5_EyeTrackingFenster - Kopie.xlsx"
WINDOWS_SHEET = "EyeTracking_Zeitfenster"

OUTPUT_FILE = PROJECT_ROOT / "Eyetracking_Auswertung_final.xlsx"


# Versuchsgruppen: Ordnername -> (Group, PreDrive, InDrive)
GROUPS = {
    "01_Kontrollgruppe":         ("G1", 0, 0),
    "02_Adaptiv_KI_Vor":         ("G2", 1, 0),
    "03_Adaptiv_KI_Während":     ("G3", 0, 1),
    "04_Adaptiv_KI_Vor&Während": ("G4", 1, 1),
}

# ASC-Dateiname weicht von der ID in den übrigen Datensätzen ab
ID_ALIASES = {
    "VAAN098": "VAAN089",
}


# Datenqualität (laut Guide): Blickdaten gültig, wenn EyesGazeQ >= 0,5
GAZE_Q_MIN = 0.5

# Personen mit weniger gültigen Blickdaten werden markiert
# (Spalte DataQualityOK), aber NICHT entfernt
MIN_VALID_PCT = 50

# Definition eines Blicks
GAP_MAX_MS = 300      # Lücken mit ungültigen Daten (z.B. Blinzeln) bis
                      # zu dieser Länge unterbrechen einen Blick nicht
MIN_GLANCE_MS = 100   # kürzere Blicke werden nicht gezählt

# Grenzwerte aus der Richtlinie
LONG_GLANCE_S = 2.0   # einzelner Blick
MAX_TASK_S = 12.0     # Gesamtblickdauer auf das Tablet je Aufgabe


# Spalten aus der ASC-Datei
COL_TIME = "Messzeitpunkt"                # ms, 0 = simulationStartedAt
COL_AUTOMATION = "Automation an_aus"
COL_GAZE_Q = "EyesGazeQ"
COL_AOI = "WorldIntersecID"               # ungefilterte AOI-Nummer
COL_POINT_X = "WorldIntersecWorldPointX"  # Schnittpunkt des Blicks
COL_POINT_Y = "WorldIntersecWorldPointY"
COL_POINT_Z = "WorldIntersecWorldPointZ"


# AOIs: Kurzname in den Spalten -> Beschreibung / Zuordnung
AOIS = {
    "Road":        "Frontscheibe / Straße (WorldIntersecID = 2)",
    "Cluster":     "Kombiinstrument hinter dem Lenkrad (ID = 7)",
    "Tablet":      "Tablet / Zentraldisplay (ID = 0 MIT Schnittpunkt, "
                   "d.h. WorldPointY oder WorldPointZ ungleich 0)",
    "RearMirror":  "Innenspiegel (ID = 5)",
    "MirrorLeft":  "Außenspiegel links (ID = 9 und WorldPointX < 0)",
    "MirrorRight": "Außenspiegel rechts (ID = 9 und WorldPointX >= 0)",
    "NoAOI":       "gültiger Blick ohne Treffer eines AOI "
                   "(ID = 0 OHNE Schnittpunkt)",
}


# ============================================================
# 2  HILFSFUNKTIONEN: DATEIEN UND IDs
# ============================================================

def nfc(text):
    """Vereinheitlicht Umlaute im Ordnernamen (Mac: "ä" = "a" + "¨")."""
    return unicodedata.normalize("NFC", text)


def participant_id(asc_file):
    """'Fahrt1_ADNA270.1.asc' -> 'ADNA270', 'BEUA003.1.asc' -> 'BEUA003'"""
    stem = re.sub(r"^Fahrt\d+_", "", asc_file.stem)   # Präfix entfernen
    stem = re.sub(r"\.\d+$", "", stem)                # ".1" / ".2" entfernen
    return ID_ALIASES.get(stem, stem)


def simulation_start(group_folder_name, pid):
    """
    simulationStartedAt aus der JSON-Datei der WebApp (nur G3/G4).
    Gibt None zurück, wenn es keine JSON-Datei / keinen Zeitstempel gibt.
    """
    for folder in TUTOR_FOLDER.iterdir():
        if nfc(folder.name) != group_folder_name:
            continue
        json_file = folder / f"{pid}.json"
        if not json_file.exists():
            return None
        with open(json_file, encoding="utf-8") as f:
            value = json.load(f).get("simulationStartedAt")
        return pd.Timestamp(value) if value else None
    return None


# ============================================================
# 3  ROHDATEN EINLESEN UND AOI PRO SAMPLE BESTIMMEN
# ============================================================

def read_samples(asc_file):
    """
    Liest die benötigten Spalten einer ASC-Datei und bestimmt für jedes
    Sample (60 Hz), ob es gültig ist und auf welche AOI geblickt wird.
    """
    df = pd.read_csv(
        asc_file,
        usecols=[COL_TIME, COL_AUTOMATION, COL_GAZE_Q, COL_AOI,
                 COL_POINT_X, COL_POINT_Y, COL_POINT_Z],
        encoding="latin-1",
    ).dropna(subset=[COL_TIME])

    aoi_id = df[COL_AOI].fillna(0).to_numpy()
    x = df[COL_POINT_X].fillna(0).to_numpy()
    y = df[COL_POINT_Y].fillna(0).to_numpy()
    z = df[COL_POINT_Z].fillna(0).to_numpy()
    has_hit = (y != 0) | (z != 0)   # Blick trifft ein Objekt

    # Gültig nur bei ausreichender Trackingqualität
    # (-999999 = keine Daten -> ebenfalls ungültig)
    valid = df[COL_GAZE_Q].fillna(0).to_numpy() >= GAZE_Q_MIN

    # AOI pro Sample (ungültige Samples bekommen "Invalid")
    aoi = np.select(
        [
            ~valid,
            aoi_id == 2,
            aoi_id == 7,
            (aoi_id == 0) & has_hit,
            aoi_id == 5,
            (aoi_id == 9) & (x < 0),
            (aoi_id == 9) & (x >= 0),
        ],
        ["Invalid", "Road", "Cluster", "Tablet", "RearMirror",
         "MirrorLeft", "MirrorRight"],
        default="NoAOI",
    )

    t_s = df[COL_TIME].to_numpy() / 1000          # Zeit in Sekunden
    dt_s = float(np.median(np.diff(t_s)))         # Abtastintervall (~1/60 s)

    return pd.DataFrame({
        "t_s": t_s,
        "valid": valid,
        "aoi": aoi,
        "automation": df[COL_AUTOMATION].fillna(0).to_numpy() > 0.5,
    }), dt_s


# ============================================================
# 4  BLICKE (GLANCES) BESTIMMEN
# ============================================================

def find_glances(is_target, is_invalid, t_s, dt_s):
    """
    Ein Blick = zusammenhängende Samples, in denen is_target zutrifft.
    Kurze Lücken (<= GAP_MAX_MS) mit ungültigen Daten werden überbrückt.
    Rückgabe: Tabelle mit Startindex, Start, Ende und Dauer jedes Blicks.
    """
    # Beginn und Ende aller zusammenhängenden Abschnitte finden
    padded = np.concatenate(([False], is_target, [False]))
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(int)) == -1)   # exklusiv

    glances = []
    for start, end in zip(starts, ends):
        if glances:
            previous_end = glances[-1][1]
            gap_s = (start - previous_end) * dt_s
            gap_only_invalid = is_invalid[previous_end:start].all()
            # Kurze Lücke aus ungültigen Daten -> mit vorherigem Blick verbinden
            if gap_s * 1000 <= GAP_MAX_MS and gap_only_invalid:
                glances[-1][1] = end
                continue
        glances.append([start, end])

    glances = pd.DataFrame(glances, columns=["i_start", "i_end"])
    if glances.empty:
        return glances.assign(start_s=[], end_s=[], dur_s=[])

    glances["start_s"] = t_s[glances["i_start"]]
    glances["end_s"] = t_s[glances["i_end"] - 1] + dt_s
    glances["dur_s"] = glances["end_s"] - glances["start_s"]

    # Sehr kurze Blicke verwerfen
    return glances[glances["dur_s"] * 1000 >= MIN_GLANCE_MS].reset_index(drop=True)


def all_glances(samples, dt_s):
    """Blicke für jede AOI und für Off-Road (alles außer Straße)."""
    aoi = samples["aoi"].to_numpy()
    t_s = samples["t_s"].to_numpy()
    invalid = aoi == "Invalid"

    glances = {name: find_glances(aoi == name, invalid, t_s, dt_s)
               for name in AOIS}

    # Off-Road-Blick: jeder gültige Blick, der nicht auf die Straße geht.
    # Wechsel zwischen Nicht-Straßen-AOIs (z.B. Kombi -> Tablet) bleiben
    # EIN Off-Road-Blick.
    off_road = (aoi != "Road") & ~invalid
    glances["OffRoad"] = find_glances(off_road, invalid, t_s, dt_s)

    return glances


# ============================================================
# 5  KENNZAHLEN FÜR EINEN ZEITABSCHNITT
# ============================================================

def compute_metrics(samples, glances, dt_s, in_segment):
    """
    Berechnet alle Kennzahlen für die Samples, bei denen in_segment True ist.
    - Prozentwerte beziehen sich auf die gültige Blickzeit (ValidEyeTimeSec)
    - Raten (pro Minute) beziehen sich auf die Abschnittsdauer
      (AnalysisTimeSec)
    - Blicke zählen zu dem Abschnitt, in dem sie beginnen
    """
    aoi = samples["aoi"].to_numpy()[in_segment]

    analysis_s = in_segment.sum() * dt_s
    valid_s = (aoi != "Invalid").sum() * dt_s

    result = {
        "AnalysisTimeSec": analysis_s,
        "ValidEyeTimeSec": valid_s,
        "ValidEyeDataPct": 100 * valid_s / analysis_s if analysis_s else np.nan,
        "AutomationPct": (100 * samples["automation"].to_numpy()[in_segment].mean()
                          if analysis_s else np.nan),
    }

    # Ohne gültige Blickdaten sind alle Blick-Kennzahlen fehlend (leer)
    if valid_s == 0:
        return result

    minutes = analysis_s / 60

    for name in ["OffRoad"] + list(AOIS):

        # Blickzeit (Samples) in diesem Abschnitt
        if name == "OffRoad":
            time_s = ((aoi != "Road") & (aoi != "Invalid")).sum() * dt_s
        else:
            time_s = (aoi == name).sum() * dt_s

        # Blicke, die in diesem Abschnitt beginnen
        g = glances[name]
        g = g[in_segment[g["i_start"]]] if len(g) else g
        n_long = int((g["dur_s"] > LONG_GLANCE_S).sum()) if len(g) else 0

        prefix = "PEORT" if name == "OffRoad" else name
        result[f"{prefix}_Pct"] = 100 * time_s / valid_s
        result[f"{name}_TimeSec"] = time_s
        result[f"{name}_Glances_N"] = len(g)
        result[f"{name}_Glances_PerMin"] = len(g) / minutes
        result[f"{name}_MeanGlanceDurSec"] = g["dur_s"].mean() if len(g) else np.nan
        result[f"{name}_MaxGlanceDurSec"] = g["dur_s"].max() if len(g) else np.nan
        result[f"{name}_GlancesOver2s_N"] = n_long
        result[f"{name}_GlancesOver2s_PerMin"] = n_long / minutes

    return result


def mask_from_windows(t_s, windows):
    """True für alle Samples, die in einem der Zeitfenster (start, end) liegen."""
    mask = np.zeros(len(t_s), dtype=bool)
    for start, end in windows:
        mask |= (t_s >= start) & (t_s < end)
    return mask


def glance_time_in_windows(glances, windows):
    """
    Summe der Blickdauern innerhalb der Zeitfenster (für die 12-s-Regel).
    Blicke über die Fenstergrenze hinaus zählen nur anteilig.
    """
    total = 0.0
    for start, end in windows:
        overlap = (np.minimum(glances["end_s"], end)
                   - np.maximum(glances["start_s"], start))
        total += overlap.clip(lower=0).sum()
    return total


# ============================================================
# 6  ZEITFENSTER DER TUTORINTERAKTIONEN
# ============================================================

def load_interaction_windows():
    """
    Liest die Zeitfenster aus der Excel-Datei.
    Rückgabe: {ID: Tabelle der Fenster mit StartUTC/EndUTC als Zeitstempel}
    """
    windows = pd.read_excel(WINDOWS_FILE, sheet_name=WINDOWS_SHEET)
    windows["StartUTC"] = pd.to_datetime(windows["StartUTC"]).dt.tz_localize("UTC")
    windows["EndUTC"] = pd.to_datetime(windows["EndUTC"]).dt.tz_localize("UTC")
    return {pid: part for pid, part in windows.groupby("Participant")}


def interaction_blocks(windows, sim_start):
    """
    Wandelt die Fenster einer Person in Sekunden seit Simulationsstart um
    (= Zeitachse der ASC-Datei) und bildet pro Interaktion (Block) zwei
    Varianten:
      Capped60  Fenster mit 60-s-Regel (wie in der Excel-Datei)
      Full      ein Fenster vom ersten Beginn bis zum letzten Ende des
                Blocks (alle Pausen innerhalb der Interaktion zählen mit)
    """
    blocks = []
    for block, part in windows.groupby("Block"):
        start = (part["StartUTC"] - sim_start).dt.total_seconds().to_numpy()
        end = (part["EndUTC"] - sim_start).dt.total_seconds().to_numpy()
        blocks.append({
            "Block": block,
            "Initiator": part["OriginalClass"].iloc[0],
            "Capped60": list(zip(start, end)),
            "Full": [(start.min(), end.max())],
        })
    return blocks


# ============================================================
# 7  HAUPTPROGRAMM
# ============================================================

def main():

    print("FINALE EYETRACKING-AUSWERTUNG")
    print("-" * 60)

    interaction_windows = load_interaction_windows()

    rows_drive = []         # Blatt Gesamtfahrt
    rows_segments = []      # Blatt Tutor_Abschnitte
    rows_interactions = []  # Blatt Tutor_Interaktionen

    for group_folder in sorted(DRIVING_FOLDER.iterdir()):

        group_name = nfc(group_folder.name)
        if group_name not in GROUPS:
            continue
        group, pre_drive, in_drive = GROUPS[group_name]

        for asc_file in sorted(group_folder.glob("*.asc")):

            pid = participant_id(asc_file)
            ids = {"ID": pid, "Group": group,
                   "PreDrive": pre_drive, "InDrive": in_drive}

            # --- Rohdaten und Blicke ---------------------------------
            samples, dt_s = read_samples(asc_file)
            glances = all_glances(samples, dt_s)
            t_s = samples["t_s"].to_numpy()

            # --- Gesamtfahrt -----------------------------------------
            whole_drive = np.ones(len(samples), dtype=bool)
            metrics = compute_metrics(samples, glances, dt_s, whole_drive)
            metrics["DataQualityOK"] = int(
                metrics["ValidEyeDataPct"] >= MIN_VALID_PCT)
            rows_drive.append({**ids, "SourceFile": asc_file.name, **metrics})

            print(f"{group} {pid:<9} gültig {metrics['ValidEyeDataPct']:5.1f} %"
                  f"  PEORT {metrics.get('PEORT_Pct', np.nan):5.1f} %")

            # --- Tutorinteraktionen (nur Personen mit Zeitfenstern) ----
            if pid not in interaction_windows:
                continue

            sim_start = simulation_start(group_name, pid)
            if sim_start is None:
                print(f"    kein simulationStartedAt für {pid} -> "
                      f"Tutorinteraktion übersprungen")
                continue

            blocks = interaction_blocks(interaction_windows[pid], sim_start)

            # Abschnitte: alle Interaktionen zusammen (beide Varianten)
            # und die Fahrt außerhalb der Interaktionen
            all_capped = [w for b in blocks for w in b["Capped60"]]
            all_full = [w for b in blocks for w in b["Full"]]
            segments = {
                "Interaction_Capped60": mask_from_windows(t_s, all_capped),
                "Interaction_Full": mask_from_windows(t_s, all_full),
                "OutsideInteraction": ~mask_from_windows(t_s, all_full),
            }

            # Prüfung der 12-s-Regel je Interaktion und Variante
            over_12s = {"Capped60": 0, "Full": 0}

            for block in blocks:
                for variant in ["Capped60", "Full"]:

                    windows = block[variant]
                    in_block = mask_from_windows(t_s, windows)
                    block_metrics = compute_metrics(
                        samples, glances, dt_s, in_block)

                    # Gesamtblickdauer auf das Tablet in dieser Interaktion
                    tablet_s = glance_time_in_windows(glances["Tablet"], windows)
                    exceeds = int(tablet_s > MAX_TASK_S)
                    over_12s[variant] += exceeds

                    rows_interactions.append({
                        **ids,
                        "Variant": variant,
                        "Block": block["Block"],
                        "Initiator": block["Initiator"],
                        "StartSec": min(s for s, _ in windows),
                        "EndSec": max(e for _, e in windows),
                        "WindowDurationSec": sum(e - s for s, e in windows),
                        "TabletTotalGlanceTimeSec": tablet_s,
                        "TabletOver12s": exceeds,
                        **block_metrics,
                    })

            for segment, mask in segments.items():
                segment_metrics = compute_metrics(samples, glances, dt_s, mask)
                extra = {}
                if segment == "Interaction_Capped60":
                    extra = {"Interactions_N": len(blocks),
                             "InteractionsOver12s_N": over_12s["Capped60"]}
                elif segment == "Interaction_Full":
                    extra = {"Interactions_N": len(blocks),
                             "InteractionsOver12s_N": over_12s["Full"]}
                rows_segments.append({**ids, "Segment": segment,
                                      **extra, **segment_metrics})

    # ========================================================
    # 8  EXCEL SCHREIBEN
    # ========================================================

    sheets = {
        "Gesamtfahrt": pd.DataFrame(rows_drive),
        "Tutor_Abschnitte": pd.DataFrame(rows_segments),
        "Tutor_Interaktionen": pd.DataFrame(rows_interactions),
    }

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

        for name, table in sheets.items():
            table.round(3).to_excel(writer, sheet_name=name, index=False,
                                    freeze_panes=(1, 1))

        variable_documentation(sheets).to_excel(
            writer, sheet_name="Variablen", index=False, freeze_panes=(1, 0))
        definitions().to_excel(
            writer, sheet_name="Definitionen", index=False)

        # Spaltenbreite an die Überschriften anpassen (nur Optik)
        for sheet in writer.sheets.values():
            for column in sheet.columns:
                width = max(len(str(c.value or "")) for c in column[:50])
                sheet.column_dimensions[column[0].column_letter].width = \
                    min(max(width + 2, 8), 80)

    print("-" * 60)
    print(f"Gespeichert: {OUTPUT_FILE}")


# ============================================================
# 9  DOKUMENTATION (Tabellenblätter "Variablen" und "Definitionen")
# ============================================================

def describe_column(column):
    """Einheit und Beschreibung einer Spalte (anhand des Namens)."""
    fixed = {
        "ID": ("–", "Versuchspersonen-ID (wie in den übrigen Datensätzen)"),
        "Group": ("–", "G1 Kontrolle, G2 nur Pre-Drive, G3 nur In-Drive, "
                       "G4 Pre- und In-Drive"),
        "PreDrive": ("0/1", "Tutor vor der Fahrt (1 = ja)"),
        "InDrive": ("0/1", "Tutor während der Fahrt (1 = ja)"),
        "SourceFile": ("–", "ausgewertete ASC-Datei"),
        "DataQualityOK": ("0/1", f"1 = ValidEyeDataPct >= {MIN_VALID_PCT} % "
                                 "(Gesamtfahrt)"),
        "Segment": ("–", "Interaction_Capped60 = alle Interaktionsfenster mit "
                         "60-s-Regel; Interaction_Full = alle Interaktionen "
                         "ohne Regel (alle Pausen innerhalb einer Interaktion); "
                         "OutsideInteraction = restliche Fahrt"),
        "Variant": ("–", "Capped60 = mit 60-s-Regel, Full = ohne Regel"),
        "Block": ("–", "Nummer der Tutorinteraktion (Aktivierung) aus der "
                       "Zeitfenster-Datei"),
        "Initiator": ("–", "Proactive oder UserInitiated"),
        "StartSec": ("s", "Beginn der Interaktion, Sekunden seit "
                          "Simulationsstart"),
        "EndSec": ("s", "Ende der Interaktion, Sekunden seit Simulationsstart"),
        "WindowDurationSec": ("s", "Summe der Zeitfenster der Interaktion"),
        "TabletTotalGlanceTimeSec": ("s", "Summe der Blickdauern auf das "
                                          "Tablet innerhalb der Zeitfenster"),
        "TabletOver12s": ("0/1", f"1 = TabletTotalGlanceTimeSec > "
                                 f"{MAX_TASK_S:g} s"),
        "Interactions_N": ("Anzahl", "Anzahl Tutorinteraktionen"),
        "InteractionsOver12s_N": ("Anzahl", f"Interaktionen mit "
                                            f"Tablet-Gesamtblickdauer > "
                                            f"{MAX_TASK_S:g} s"),
        "AnalysisTimeSec": ("s", "Dauer des betrachteten Abschnitts"),
        "ValidEyeTimeSec": ("s", f"Zeit mit gültigen Blickdaten "
                                 f"(EyesGazeQ >= {GAZE_Q_MIN})"),
        "ValidEyeDataPct": ("%", "ValidEyeTimeSec / AnalysisTimeSec"),
        "AutomationPct": ("%", "Zeitanteil mit aktiver Automation, bezogen "
                               "auf AnalysisTimeSec"),
        "PEORT_Pct": ("%", "Percent Eyes-Off-Road Time: gültige Blickzeit "
                           "nicht auf der Straße / ValidEyeTimeSec"),
    }
    if column in fixed:
        return fixed[column]

    name, _, metric = column.partition("_")
    target = "Off-Road (alle gültigen Blicke außer Straße)" \
        if name == "OffRoad" else f"AOI {name}"
    metrics = {
        "Pct": ("%", f"Blickzeit {target} / ValidEyeTimeSec"),
        "TimeSec": ("s", f"Blickzeit {target}"),
        "Glances_N": ("Anzahl", f"Anzahl Blicke {target}"),
        "Glances_PerMin": ("1/min", f"Blicke {target} pro Minute "
                                    f"AnalysisTimeSec"),
        "MeanGlanceDurSec": ("s", f"mittlere Blickdauer {target} "
                                  f"(leer, wenn kein Blick)"),
        "MaxGlanceDurSec": ("s", f"längster Blick {target} "
                                 f"(leer, wenn kein Blick)"),
        "GlancesOver2s_N": ("Anzahl", f"Blicke {target} länger als "
                                      f"{LONG_GLANCE_S:g} s"),
        "GlancesOver2s_PerMin": ("1/min", f"Blicke {target} > "
                                          f"{LONG_GLANCE_S:g} s pro Minute "
                                          f"AnalysisTimeSec"),
    }
    return metrics.get(metric, ("", ""))


def variable_documentation(sheets):
    """Eine Zeile pro Spalte und Tabellenblatt: Einheit und Beschreibung."""
    rows = []
    for sheet, table in sheets.items():
        for column in table.columns:
            unit, description = describe_column(column)
            rows.append({"Sheet": sheet, "Variable": column,
                         "Unit": unit, "Description": description})
    return pd.DataFrame(rows)


def definitions():
    """Definitionen der AOIs, Zeiträume und Parameter."""
    rows = [("AOI", name, text) for name, text in AOIS.items()]
    rows += [
        ("AOI", "OffRoad", "alle gültigen Blicke außer Road "
                           "(inkl. NoAOI)"),
        ("Datenqualität", "Gültige Blickdaten",
         f"EyesGazeQ >= {GAZE_Q_MIN}; alle anderen Samples (inkl. "
         f"-999999 = keine Aufzeichnung) gelten als ungültig"),
        ("Datenqualität", "Bezugsgrößen",
         "Prozentwerte (_Pct) beziehen sich auf ValidEyeTimeSec, "
         "Raten (_PerMin) auf AnalysisTimeSec"),
        ("Datenqualität", "Fehlende Werte",
         "leer = nicht berechenbar (z.B. keine gültigen Blickdaten, kein "
         "Blick für Mittelwert); 0 = tatsächlich 0"),
        ("Blick", "Definition",
         f"zusammenhängende gültige Samples auf derselben AOI; Lücken mit "
         f"ungültigen Daten bis {GAP_MAX_MS} ms werden überbrückt; Blicke "
         f"< {MIN_GLANCE_MS} ms werden verworfen; ein Blick zählt zu dem "
         f"Abschnitt, in dem er beginnt"),
        ("Blick", "Off-Road-Blick",
         "zusammenhängende Zeit ohne Blick auf die Straße; Wechsel "
         "zwischen anderen AOIs bleiben ein Blick"),
        ("Zeitraum", "Gesamtfahrt",
         "gesamte ASC-Aufzeichnung (Zeit 0 = simulationStartedAt)"),
        ("Zeitraum", "Interaction_Capped60",
         f"Zeitfenster aus {WINDOWS_FILE.name} (Blatt {WINDOWS_SHEET}): "
         f"Nachrichtenabstände <= 60 s vollständig, längere Abstände nur "
         f"30 s nach und 30 s vor den Nachrichten"),
        ("Zeitraum", "Interaction_Full",
         "je Interaktion vom ersten Fensterbeginn bis zum letzten "
         "Fensterende (alle Pausen innerhalb einer Interaktion zählen)"),
        ("Zeitraum", "OutsideInteraction",
         "Gesamtfahrt ohne Interaction_Full"),
        ("Grenzwert", "2-s-Regel", f"einzelner Blick > {LONG_GLANCE_S:g} s"),
        ("Grenzwert", "12-s-Regel",
         f"Summe der Tablet-Blickdauern innerhalb einer Interaktion "
         f"> {MAX_TASK_S:g} s"),
        ("ID", "VAAN098 / VAAN089",
         "ASC-Datei heißt VAAN098, in allen anderen Datensätzen VAAN089 -> "
         "ID VAAN089 verwendet"),
    ]
    return pd.DataFrame(rows, columns=["Category", "Name", "Definition"])


# ============================================================
# START
# ============================================================

def analyze(config_name: str) -> None:
    raise NotImplementedError(
        "The eye-tracking analysis has not yet been migrated to prepared Parquet data. "
        "Run 'lfecli prepare' for now."
    )


if __name__ == "__main__":
    main()
