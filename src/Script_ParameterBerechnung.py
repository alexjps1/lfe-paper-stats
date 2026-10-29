from pathlib import Path
import re
import time
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# KONFIGURATION
# ============================================================

# Projektordner = Ordner, in dem dieses Script liegt
PROJECT_ROOT = Path(__file__).resolve().parent

DRIVING_FOLDER = PROJECT_ROOT / "Fahrdaten_Eyetracking"
OUTPUT_FOLDER = PROJECT_ROOT / "02_Parameter_Eyetracking"


# Spalten in der ASC-Datei
# Bewusst die UNGEFILTERTEN Spalten: in den gefilterten Spalten
# (WorldIntersecFilter...) ist der Schnittpunkt in manchen Dateien auch
# ohne Treffer ungleich 0 -> Zentraldisplay wäre nicht erkennbar.
TIME_COLUMN = "Messzeitpunkt"
AUTOMATION_COLUMN = "Automation an_aus"
GAZE_Q_COLUMN = "EyesGazeQ"
AOI_COLUMN = "WorldIntersecID"
POINT_COLUMNS = [
    "WorldIntersecWorldPointX",
    "WorldIntersecWorldPointY",
    "WorldIntersecWorldPointZ",
]


# ------------------------------------------------------------
# Zuordnung AOI-Nummer -> AOI im Smart-Eye-Weltmodell
# ------------------------------------------------------------
# Aus den Daten abgeleitet (Blickrichtung, Schnittpunkt-Koordinaten
# und Blickverhalten während der KI-Gespräche):
#
#   2  Frontscheibe        geradeaus
#   7  Kombiinstrument     geradeaus, ca. 18° unten
#   5  Innenspiegel        ca. 30° rechts, auf Augenhöhe, nah am Fahrer
#   9  Außenspiegel        links (X < 0) UND rechts (X > 0) – gleiche Nummer,
#                          Seite wird über die X-Koordinate getrennt
#   0  Zentraldisplay      ca. 22° rechts, 25° unten – NUR wenn ein
#                          Schnittpunkt existiert (Y oder Z ungleich 0).
#                          Blickanteil steigt in den Gruppen 03/04
#                          während der KI-Gespräche deutlich an.
#   0  ohne Schnittpunkt   (Y = Z = 0): kein Objekt getroffen
#                          -> "kein_AOI" bei guter Trackingqualität,
#                          sonst Trackingverlust
#   -999999                keine Eyetracking-Daten aufgezeichnet
AOI_LISTE = [
    "frontscheibe",
    "kombiinstrument",
    "zentraldisplay",
    "innenspiegel",
    "aussenspiegel_links",
    "aussenspiegel_rechts",
    "kein_aoi",
]
AOI_STRASSE = "frontscheibe"
UNGUELTIG = "kein_tracking"


# Blick ohne AOI-Treffer zählt als gültig, wenn EyesGazeQ >= Schwelle
GAZE_Q_MIN = 0.5


# Datenqualität: Probanden mit weniger gültigen Blickdaten werden
# NICHT gelöscht, sondern in der Spalte "datenqualitaet_ok" markiert
MIN_GUELTIG_ANTEIL = 0.5


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def nfc(text):
    """
    Vereinheitlicht Umlaute (Mac speichert "ä" teils als "a" + "¨").
    """

    return unicodedata.normalize("NFC", text)


def participant_id_from_file(file):
    """
    "Fahrt1_ADNA270.1.asc" -> "ADNA270", "BEUA003.1.asc" -> "BEUA003"
    """

    stem = re.sub(r"^Fahrt\d+_", "", file.stem)
    return re.sub(r"\.\d+$", "", stem)


def find_asc_files():
    """
    Liefert eine Liste (Gruppe, Proband, Datei) für alle ASC-Dateien.
    """

    files = []

    for group_folder in sorted(DRIVING_FOLDER.iterdir()):

        if not group_folder.is_dir():
            continue

        for file in sorted(group_folder.glob("*.asc")):
            files.append((
                nfc(group_folder.name),
                participant_id_from_file(file),
                file,
            ))

    return files


def classify_aoi(df):
    """
    Ordnet jedem Sample eine AOI aus AOI_LISTE zu (oder UNGUELTIG).
    """

    aoi_id = df[AOI_COLUMN].fillna(0).to_numpy(float)
    x, y, z = (df[c].fillna(0).to_numpy(float) for c in POINT_COLUMNS)
    gaze_q = df[GAZE_Q_COLUMN].fillna(0).to_numpy(float)

    has_hit = (y != 0) | (z != 0)

    unknown = ~np.isin(aoi_id, [0, 2, 5, 7, 9]) & (aoi_id > 0)
    if unknown.any():
        print(
            f"    WARNUNG: unbekannte AOI-Nummern "
            f"{sorted(set(aoi_id[unknown].astype(int)))} -> ungültig"
        )

    return np.select(
        [
            aoi_id == 2,
            aoi_id == 7,
            (aoi_id == 0) & has_hit,
            aoi_id == 5,
            (aoi_id == 9) & (x < 0),
            (aoi_id == 9) & (x >= 0),
            (aoi_id == 0) & ~has_hit & (gaze_q >= GAZE_Q_MIN),
        ],
        AOI_LISTE,
        default=UNGUELTIG,
    )


def participant_parameters(file):
    """
    Berechnet Blickzeit pro AOI und Eyes-off-Road für eine ASC-Datei.

    Prozentwerte "_prozent" beziehen sich auf die gültige Blickzeit
    (alle Samples außer Trackingverlust).
    Eyes-off-Road = gültige Blickzeit, die nicht auf die Frontscheibe fällt.
    """

    df = pd.read_csv(
        file,
        usecols=[
            TIME_COLUMN, AUTOMATION_COLUMN, GAZE_Q_COLUMN, AOI_COLUMN,
            *POINT_COLUMNS,
        ],
        encoding="latin-1",
    ).dropna(subset=[TIME_COLUMN])

    dt_s = float(np.median(np.diff(df[TIME_COLUMN].to_numpy(float)))) / 1000
    aoi = pd.Series(classify_aoi(df))

    n_total = len(aoi)
    counts = aoi.value_counts()
    n_valid = n_total - int(counts.get(UNGUELTIG, 0))
    n_road = int(counts.get(AOI_STRASSE, 0))
    n_off_road = n_valid - n_road

    def percent(n):
        return 100 * n / n_valid if n_valid else np.nan

    result = {
        "eyes_off_road_prozent": percent(n_off_road),
    }

    for name in AOI_LISTE:
        result[f"{name}_prozent"] = percent(int(counts.get(name, 0)))

    result["eyes_off_road_s"] = n_off_road * dt_s

    for name in AOI_LISTE:
        result[f"{name}_s"] = int(counts.get(name, 0)) * dt_s

    result.update({
        "fahrtdauer_s": n_total * dt_s,
        "gueltige_blickzeit_s": n_valid * dt_s,
        "anteil_gueltig_prozent": 100 * n_valid / n_total,
        "anteil_kein_tracking_prozent": 100 * (n_total - n_valid) / n_total,
        "anteil_automation_prozent": (
            100 * (df[AUTOMATION_COLUMN].fillna(0) > 0.5).mean()
        ),
    })

    return result


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EYETRACKING – EYES-OFF-ROAD UND BLICKANTEILE PRO AOI")
    print("=" * 70)
    print()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    rows = []
    files = find_asc_files()

    # --------------------------------------------------------
    # Parameter pro Proband
    # --------------------------------------------------------

    for i, (group, participant_id, file) in enumerate(files, start=1):

        started = time.time()

        try:
            result = participant_parameters(file)
        except Exception as e:
            print(f"  FEHLER bei {file.name}: {e}")
            continue

        rows.append({
            "gruppe": group,
            "proband": participant_id,
            "datei": file.name,
            "datenqualitaet_ok": (
                result["anteil_gueltig_prozent"] >= 100 * MIN_GUELTIG_ANTEIL
            ),
            **result,
        })

        print(
            f"  [{i}/{len(files)}] {group} / {participant_id}: "
            f"Eyes-off-Road {result['eyes_off_road_prozent']:.1f} %, "
            f"Zentraldisplay {result['zentraldisplay_prozent']:.1f} %, "
            f"gültig {result['anteil_gueltig_prozent']:.1f} % "
            f"({time.time() - started:.1f} s)"
        )

    participants = pd.DataFrame(rows).round(3)

    participants_file = OUTPUT_FOLDER / "parameter_pro_proband.csv"
    participants.to_csv(participants_file, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Übersicht pro Gruppe (nur Probanden mit ausreichender Datenqualität)
    # --------------------------------------------------------

    percent_columns = ["eyes_off_road_prozent"] + [
        f"{name}_prozent" for name in AOI_LISTE
    ]

    included = participants[participants["datenqualitaet_ok"]]

    summary_rows = []

    for group, part in included.groupby("gruppe"):
        for column in percent_columns:
            values = part[column]
            summary_rows.append({
                "gruppe": group,
                "parameter": column,
                "n": int(values.count()),
                "mittelwert": values.mean(),
                "sd": values.std(ddof=1),
                "median": values.median(),
                "minimum": values.min(),
                "maximum": values.max(),
            })

    summary = pd.DataFrame(summary_rows).round(3)
    summary_file = OUTPUT_FOLDER / "uebersicht_pro_gruppe.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Konsole
    # --------------------------------------------------------

    excluded = participants[~participants["datenqualitaet_ok"]]

    if len(excluded):
        print()
        print(
            f"DATENQUALITÄT UNZUREICHEND (< {100 * MIN_GUELTIG_ANTEIL:.0f} % "
            f"gültig, in Gruppenübersicht nicht enthalten):"
        )
        for _, row in excluded.iterrows():
            print(
                f"  {row['gruppe']} / {row['proband']}: "
                f"{row['anteil_gueltig_prozent']:.1f} %"
            )

    print()
    print("MITTELWERTE PRO GRUPPE [% der gültigen Blickzeit]:")
    print()

    table = summary.pivot(
        index="parameter", columns="gruppe", values="mittelwert"
    ).reindex(percent_columns)

    with pd.option_context("display.width", 200):
        print(table.round(2).to_string())

    print()
    print("=" * 70)
    print("FERTIG")
    print("=" * 70)
    print(f"Pro Proband:  {participants_file}")
    print(f"Pro Gruppe:   {summary_file}")
    print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
