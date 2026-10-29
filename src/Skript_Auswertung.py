from pathlib import Path
from itertools import combinations
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# KONFIGURATION
# ============================================================

# Projektordner = Ordner, in dem dieses Script liegt
PROJECT_ROOT = Path(__file__).resolve().parent

DRIVING_FOLDER = PROJECT_ROOT / "Fahrdaten_Eyetracking"
OUTPUT_FOLDER = PROJECT_ROOT / "01_Auswertung_Eyetracking"
KENNWERTE_FILE = OUTPUT_FOLDER / "kennwerte_pro_proband.csv"


# True:  alle ASC-Dateien neu einlesen (dauert einige Minuten)
# False: vorhandene kennwerte_pro_proband.csv verwenden
#        (nur Statistik + Grafiken neu rechnen)
KENNWERTE_NEU_BERECHNEN = True


# Kurznamen der Gruppen für Tabellen und Grafiken
GRUPPEN_LABEL = {
    "01_Kontrollgruppe": "Kontrolle",
    "02_Adaptiv_KI_Vor": "KI vor",
    "03_Adaptiv_KI_Während": "KI während",
    "04_Adaptiv_KI_Vor&Während": "KI vor & während",
}


# Areas of Interest (Spalte WorldIntersecFilterID).
# 0 = kein Schnittpunkt / kein Tracking.
# ACHTUNG: Die Bedeutung der IDs ist aus der Blickrichtung geschätzt
# und muss mit der SILAB-/Smart-Eye-Konfiguration abgeglichen werden!
AOI_STRASSE = 2
AOI_NAMEN = {
    2: "Straße / Frontscheibe (?)",
    5: "rechts oben – Innenspiegel (?)",
    7: "unten mittig – Kombiinstrument (?)",
    9: "links – Außenspiegel (?)",
}


# Analyse-Abschnitte: Gesamtfahrt sowie getrennt nach Automationsstatus
# (Spalte "Automation an_aus")
ABSCHNITTE = ["gesamt", "manuell", "automatisiert"]


# Schwellenwerte Datenqualität
GAZE_Q_MIN = 0.5            # EyesGazeQ: Blickdaten gültig ab
PUPIL_Q_MIN = 0.2           # EyeyPupilDiameterQ: Pupillendaten gültig ab
LID_Q_MIN = 0.5             # EyesLidOpeningQ: Lidöffnung gültig ab
MIN_GUELTIG_ANTEIL = 0.5    # Proband wird ausgeschlossen, wenn weniger
                            # gültige Blickdaten (Gesamtfahrt)
MIN_ABSCHNITT_MIN = 1.0     # Abschnitt kürzer -> keine Kennwerte
MIN_GUELTIGE_SAMPLES = 600  # (= 10 s bei 60 Hz) pro Kennwert


# Ereignis-Parameter
BLINK_MAX_MS = 1000         # längere "Blinzler" = Artefakt
FIXATION_VT_DEG_S = 30      # I-VT: Geschwindigkeit < Schwelle = Fixation
FIXATION_MIN_MS = 100       # Mindestdauer einer Fixation
BLICK_MIN_MS = 100          # Mindestdauer eines Blicks auf eine AOI
LUECKE_FUELLEN_MS = 300     # Tracking-Lücken (ID 0) bis zu dieser Länge
                            # werden mit der vorherigen AOI aufgefüllt
LANGER_BLICK_S = 2.0        # Blicke abseits der Straße länger als ...


# Farben der Gruppen (validierte kategoriale Palette, feste Reihenfolge)
GRUPPEN_FARBEN = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


# Spalten, die aus der ASC-Datei gelesen werden
TIME_COL = "Messzeitpunkt"
ASC_COLUMNS = [
    TIME_COL,
    "Automation an_aus",
    "EyesGazeQ",
    "EyesFilterGazeDirX",
    "EyesFilterGazeDirY",
    "EyesFilterGazeDirZ",
    "EyesBlink",
    "EyesPupilDiameter",
    "EyeyPupilDiameterQ",
    "EyesLidOpening",
    "EyesLidOpeningQ",
    "WorldIntersecFilterID",
]


# ============================================================
# KENNWERTE (abhängige Variablen)
# ============================================================

def kennwert_liste():
    """
    Spaltenname -> Beschriftung aller Kennwerte, die statistisch
    verglichen werden.
    """

    kennwerte = {
        "blinzelrate_pro_min": "Blinzelrate [1/min]",
        "blinzeldauer_ms": "Blinzeldauer [ms]",
        "pupille_mm": "Pupillendurchmesser [mm]",
        "lidoeffnung_mm": "Lidöffnung [mm]",
        "blick_sd_horizontal_grad": "Blickstreuung horizontal [°]",
        "blick_sd_vertikal_grad": "Blickstreuung vertikal [°]",
        "fixationsrate_pro_min": "Fixationen [1/min]",
        "fixationsdauer_ms": "Fixationsdauer [ms]",
    }

    for aoi_id, name in AOI_NAMEN.items():
        kennwerte[f"anteil_aoi{aoi_id}_prozent"] = f"Blickanteil {name} [%]"

        if aoi_id != AOI_STRASSE:
            kennwerte[f"blicke_aoi{aoi_id}_pro_min"] = \
                f"Blicke {name} [1/min]"
            kennwerte[f"blickdauer_aoi{aoi_id}_s"] = \
                f"Blickdauer {name} [s]"

    kennwerte["blicke_abseits_pro_min"] = "Blicke abseits der Straße [1/min]"
    kennwerte["blickdauer_abseits_s"] = "Blickdauer abseits der Straße [s]"
    kennwerte["lange_blicke_abseits_pro_min"] = \
        f"Blicke abseits > {LANGER_BLICK_S:g} s [1/min]"
    kennwerte["anteil_automation_prozent"] = "Zeitanteil Automation [%]"

    return kennwerte


# ============================================================
# HILFSFUNKTIONEN – DATEIEN
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


# ============================================================
# HILFSFUNKTIONEN – EREIGNISSE
# ============================================================

def value_runs(values):
    """
    Zerlegt ein Array in Abschnitte gleicher Werte.
    Gibt (Startindex, Länge, Wert) als Arrays zurück.
    """

    values = np.asarray(values)

    if len(values) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, values

    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [len(values)])))

    return starts, lengths, values[starts]


def true_runs(mask):
    """
    Startindex und Länge aller zusammenhängenden True-Abschnitte.
    """

    starts, lengths, run_values = value_runs(np.asarray(mask, dtype=bool))
    keep = run_values
    return starts[keep], lengths[keep]


def compute_sample_data(df):
    """
    Berechnet aus den Rohdaten alle Größen pro Sample sowie die
    Ereignisse (Blinzler, Fixationen, Blicke) für die gesamte Fahrt.
    """

    t_ms = df[TIME_COL].to_numpy(float)
    dt_ms = float(np.median(np.diff(t_ms)))

    gaze_valid = df["EyesGazeQ"].to_numpy(float) >= GAZE_Q_MIN

    # ---------------- Blinzler ----------------
    # EyesBlink enthält während eines Blinzlers dessen laufende Nummer

    blink = df["EyesBlink"].fillna(0).to_numpy(float)
    starts, lengths, run_values = value_runs(blink)
    is_blink_run = (run_values > 0) & (lengths * dt_ms <= BLINK_MAX_MS)

    blinks = pd.DataFrame({
        "start": starts[is_blink_run],
        "dauer_ms": lengths[is_blink_run] * dt_ms,
    })

    in_blink = blink > 0

    # ---------------- Blickwinkel ----------------

    direction = df[[
        "EyesFilterGazeDirX",
        "EyesFilterGazeDirY",
        "EyesFilterGazeDirZ",
    ]].to_numpy(float)

    norm = np.linalg.norm(direction, axis=1)
    gaze_valid &= norm > 0.5
    unit = direction / np.where(norm > 0, norm, 1)[:, None]

    # Blickrichtung zeigt nach vorne in -Z
    yaw = np.degrees(np.arctan2(unit[:, 0], -unit[:, 2]))
    pitch = np.degrees(np.arctan2(
        unit[:, 1], np.hypot(unit[:, 0], unit[:, 2])
    ))

    # ---------------- Fixationen (I-VT) ----------------

    dot = np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1, 1)
    step_ms = np.diff(t_ms)
    velocity = np.full(len(t_ms), np.nan)
    velocity[1:] = np.degrees(np.arccos(dot)) / (step_ms / 1000)

    velocity_valid = np.zeros(len(t_ms), dtype=bool)
    velocity_valid[1:] = (
        gaze_valid[1:] & gaze_valid[:-1] & (step_ms < 2 * dt_ms)
    )

    fix_starts, fix_lengths = true_runs(
        velocity_valid & (velocity < FIXATION_VT_DEG_S)
    )
    keep = fix_lengths * dt_ms >= FIXATION_MIN_MS

    fixations = pd.DataFrame({
        "start": fix_starts[keep],
        "dauer_ms": fix_lengths[keep] * dt_ms,
    })

    # ---------------- AOI-Blicke ----------------

    gap_samples = int(round(LUECKE_FUELLEN_MS / dt_ms))

    aoi = (
        df["WorldIntersecFilterID"]
        .replace(0, np.nan)
        .ffill(limit=gap_samples)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    aoi_starts, aoi_lengths, aoi_values = value_runs(aoi)
    keep = aoi_lengths * dt_ms >= BLICK_MIN_MS

    aoi_glances = pd.DataFrame({
        "start": aoi_starts[keep],
        "dauer_s": aoi_lengths[keep] * dt_ms / 1000,
        "aoi": aoi_values[keep],
    })

    off_starts, off_lengths = true_runs(
        (aoi != AOI_STRASSE) & (aoi != 0)
    )
    keep = off_lengths * dt_ms >= BLICK_MIN_MS

    offroad_glances = pd.DataFrame({
        "start": off_starts[keep],
        "dauer_s": off_lengths[keep] * dt_ms / 1000,
    })

    return {
        "dt_ms": dt_ms,
        "automation": df["Automation an_aus"].fillna(0).to_numpy() > 0.5,
        "gaze_valid": gaze_valid,
        "in_blink": in_blink,
        "yaw": yaw,
        "pitch": pitch,
        "pupil_mm": df["EyesPupilDiameter"].to_numpy(float) * 1000,
        "pupil_valid": (
            (df["EyeyPupilDiameterQ"].to_numpy(float) >= PUPIL_Q_MIN)
            & (df["EyesPupilDiameter"].to_numpy(float) > 0)
            & ~in_blink
        ),
        "lid_mm": df["EyesLidOpening"].to_numpy(float) * 1000,
        "lid_valid": (
            (df["EyesLidOpeningQ"].to_numpy(float) >= LID_Q_MIN)
            & ~in_blink
        ),
        "aoi_raw": df["WorldIntersecFilterID"].fillna(0).to_numpy(int),
        "blinks": blinks,
        "fixations": fixations,
        "aoi_glances": aoi_glances,
        "offroad_glances": offroad_glances,
    }


def segment_metrics(data, mask):
    """
    Berechnet alle Kennwerte für die Samples in mask.
    """

    def mean_if_enough(values, valid):
        selected = values[mask & valid]
        if len(selected) < MIN_GUELTIGE_SAMPLES:
            return np.nan
        return float(np.mean(selected))

    def sd_if_enough(values, valid):
        selected = values[mask & valid]
        if len(selected) < MIN_GUELTIGE_SAMPLES:
            return np.nan
        return float(np.std(selected, ddof=1))

    def events_in_segment(events):
        return events[mask[events["start"].to_numpy(int)]]

    n_samples = int(mask.sum())
    minutes = n_samples * data["dt_ms"] / 60000
    gaze_valid = data["gaze_valid"]

    result = {
        "dauer_min": minutes,
        "anteil_gueltig_prozent": (
            100 * gaze_valid[mask].mean() if n_samples else np.nan
        ),
        "anteil_automation_prozent": (
            100 * data["automation"][mask].mean() if n_samples else np.nan
        ),
    }

    if minutes < MIN_ABSCHNITT_MIN:
        return result

    # Blinzler
    blinks = events_in_segment(data["blinks"])
    result["blinzelrate_pro_min"] = len(blinks) / minutes
    result["blinzeldauer_ms"] = (
        blinks["dauer_ms"].mean() if len(blinks) else np.nan
    )

    # Pupille / Lid
    result["pupille_mm"] = mean_if_enough(
        data["pupil_mm"], data["pupil_valid"]
    )
    result["lidoeffnung_mm"] = mean_if_enough(
        data["lid_mm"], data["lid_valid"]
    )

    # Blickstreuung
    result["blick_sd_horizontal_grad"] = sd_if_enough(
        data["yaw"], gaze_valid
    )
    result["blick_sd_vertikal_grad"] = sd_if_enough(
        data["pitch"], gaze_valid
    )

    # Fixationen
    fixations = events_in_segment(data["fixations"])
    result["fixationsrate_pro_min"] = len(fixations) / minutes
    result["fixationsdauer_ms"] = (
        fixations["dauer_ms"].mean() if len(fixations) else np.nan
    )

    # AOI-Anteile (bezogen auf alle Samples mit Blick auf eine AOI)
    aoi_raw = data["aoi_raw"][mask]
    n_on_aoi = int((aoi_raw != 0).sum())

    for aoi_id in AOI_NAMEN:
        result[f"anteil_aoi{aoi_id}_prozent"] = (
            100 * (aoi_raw == aoi_id).sum() / n_on_aoi
            if n_on_aoi >= MIN_GUELTIGE_SAMPLES else np.nan
        )

    # Blicke auf einzelne AOIs
    glances = events_in_segment(data["aoi_glances"])

    for aoi_id in AOI_NAMEN:

        if aoi_id == AOI_STRASSE:
            continue

        aoi_glances = glances[glances["aoi"] == aoi_id]
        result[f"blicke_aoi{aoi_id}_pro_min"] = len(aoi_glances) / minutes
        result[f"blickdauer_aoi{aoi_id}_s"] = (
            aoi_glances["dauer_s"].mean() if len(aoi_glances) else np.nan
        )

    # Blicke abseits der Straße
    offroad = events_in_segment(data["offroad_glances"])
    result["blicke_abseits_pro_min"] = len(offroad) / minutes
    result["blickdauer_abseits_s"] = (
        offroad["dauer_s"].mean() if len(offroad) else np.nan
    )
    result["lange_blicke_abseits_pro_min"] = (
        (offroad["dauer_s"] > LANGER_BLICK_S).sum() / minutes
    )

    return result


def compute_kennwerte():
    """
    Liest alle ASC-Dateien und berechnet die Kennwerte
    pro Proband und Abschnitt.
    """

    rows = []
    files = find_asc_files()

    for i, (group, participant_id, file) in enumerate(files, start=1):

        started = time.time()

        try:
            df = pd.read_csv(
                file,
                usecols=ASC_COLUMNS,
                encoding="latin-1",
                low_memory=False,
            )
        except Exception as e:
            print(f"  FEHLER beim Lesen von {file.name}: {e}")
            continue

        df = df.dropna(subset=[TIME_COL])
        data = compute_sample_data(df)

        masks = {
            "gesamt": np.ones(len(df), dtype=bool),
            "manuell": ~data["automation"],
            "automatisiert": data["automation"],
        }

        for segment in ABSCHNITTE:
            rows.append({
                "gruppe": group,
                "proband": participant_id,
                "datei": file.name,
                "abschnitt": segment,
                **segment_metrics(data, masks[segment]),
            })

        print(
            f"  [{i}/{len(files)}] {group} / {participant_id}: "
            f"{len(df)} Zeilen, {time.time() - started:.1f} s"
        )

    return pd.DataFrame(rows)


# ============================================================
# STATISTIK
# ============================================================

def welch_anova(samples):
    """
    Welch-ANOVA (robust bei ungleichen Varianzen).
    Gibt (F, df1, df2, p) zurück.
    """

    k = len(samples)
    n = np.array([len(s) for s in samples], dtype=float)
    means = np.array([np.mean(s) for s in samples])
    variances = np.array([np.var(s, ddof=1) for s in samples])

    if np.any(variances == 0):
        return np.nan, np.nan, np.nan, np.nan

    w = n / variances
    grand_mean = np.sum(w * means) / np.sum(w)

    a = np.sum(w * (means - grand_mean) ** 2) / (k - 1)
    tmp = np.sum((1 - w / np.sum(w)) ** 2 / (n - 1))
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp

    f_value = a / b
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * tmp)

    return f_value, df1, df2, stats.f.sf(f_value, df1, df2)


def eta_squared(samples):
    """
    Effektstärke η² der einfaktoriellen ANOVA.
    """

    values = np.concatenate(samples)
    grand_mean = values.mean()

    ss_between = sum(
        len(s) * (np.mean(s) - grand_mean) ** 2 for s in samples
    )
    ss_total = np.sum((values - grand_mean) ** 2)

    return ss_between / ss_total if ss_total > 0 else np.nan


def games_howell(samples, labels):
    """
    Post-hoc-Test nach Games-Howell (passend zur Welch-ANOVA).
    """

    k = len(samples)
    rows = []

    for i, j in combinations(range(k), 2):

        a, b = samples[i], samples[j]
        va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)

        diff = np.mean(a) - np.mean(b)
        se = np.sqrt(va + vb)

        if se == 0:
            continue

        df = (va + vb) ** 2 / (
            va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)
        )
        q = abs(diff) / se * np.sqrt(2)

        rows.append({
            "gruppe_a": labels[i],
            "gruppe_b": labels[j],
            "differenz_mittelwert": diff,
            "p_games_howell": stats.studentized_range.sf(q, k, df),
        })

    return rows


def holm(p_values):
    """
    Holm-Korrektur für multiples Testen (NaN werden ignoriert).
    """

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(~np.isnan(p))

    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p[valid])]
    m = len(order)
    running_max = 0.0

    for rank, index in enumerate(order):
        running_max = max(running_max, (m - rank) * p[index])
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def run_statistics(kennwerte):
    """
    Deskriptive Statistik, Voraussetzungen, ANOVA, Welch-ANOVA,
    Kruskal-Wallis und Post-hoc-Tests pro Abschnitt und Kennwert.
    """

    descriptive_rows = []
    test_rows = []
    posthoc_rows = []

    included = kennwerte[~kennwerte["ausgeschlossen"]]
    groups = sorted(included["gruppe"].unique())

    for segment in ABSCHNITTE:

        segment_data = included[included["abschnitt"] == segment]
        segment_tests = []

        for metric, label in kennwert_liste().items():

            if metric not in segment_data:
                continue

            # Automationsanteil ist in diesen Abschnitten konstant
            if metric == "anteil_automation_prozent" and segment != "gesamt":
                continue

            samples, sample_groups = [], []

            for group in groups:

                values = segment_data.loc[
                    segment_data["gruppe"] == group, metric
                ].dropna().to_numpy(float)

                descriptive_rows.append({
                    "abschnitt": segment,
                    "kennwert": metric,
                    "beschriftung": label,
                    "gruppe": group,
                    "n": len(values),
                    "mittelwert": np.mean(values) if len(values) else np.nan,
                    "sd": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                    "median": np.median(values) if len(values) else np.nan,
                    "q1": np.percentile(values, 25) if len(values) else np.nan,
                    "q3": np.percentile(values, 75) if len(values) else np.nan,
                    "shapiro_p": (
                        stats.shapiro(values).pvalue
                        if len(values) >= 3 and np.ptp(values) > 0
                        else np.nan
                    ),
                })

                if len(values) >= 3:
                    samples.append(values)
                    sample_groups.append(group)

            if len(samples) < 2:
                continue

            anova = stats.f_oneway(*samples)
            welch_f, welch_df1, welch_df2, welch_p = welch_anova(samples)
            kruskal = stats.kruskal(*samples)
            levene = stats.levene(*samples, center="median")
            n_total = sum(len(s) for s in samples)

            segment_tests.append({
                "abschnitt": segment,
                "kennwert": metric,
                "beschriftung": label,
                "gruppen": " | ".join(sample_groups),
                "n_gesamt": n_total,
                "levene_p": levene.pvalue,
                "anova_F": anova.statistic,
                "anova_df1": len(samples) - 1,
                "anova_df2": n_total - len(samples),
                "anova_p": anova.pvalue,
                "eta2": eta_squared(samples),
                "welch_F": welch_f,
                "welch_df1": welch_df1,
                "welch_df2": welch_df2,
                "welch_p": welch_p,
                "kruskal_H": kruskal.statistic,
                "kruskal_p": kruskal.pvalue,
                "epsilon2": kruskal.statistic / (n_total - 1),
            })

            tukey = stats.tukey_hsd(*samples)

            for row in games_howell(samples, sample_groups):

                i = sample_groups.index(row["gruppe_a"])
                j = sample_groups.index(row["gruppe_b"])

                posthoc_rows.append({
                    "abschnitt": segment,
                    "kennwert": metric,
                    "beschriftung": label,
                    **row,
                    "p_tukey": tukey.pvalue[i, j],
                })

        # Holm-Korrektur über alle Kennwerte eines Abschnitts
        if segment_tests:
            for test in ["anova", "welch", "kruskal"]:
                adjusted = holm([row[f"{test}_p"] for row in segment_tests])
                for row, p in zip(segment_tests, adjusted):
                    row[f"{test}_p_holm"] = p

        test_rows.extend(segment_tests)

    return (
        pd.DataFrame(descriptive_rows),
        pd.DataFrame(test_rows),
        pd.DataFrame(posthoc_rows),
    )


# ============================================================
# GRAFIKEN
# ============================================================

def format_p(p):
    if pd.isna(p):
        return "p = –"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def plot_boxplots(kennwerte, tests, segment, output_file):
    """
    Boxplots aller Kennwerte eines Abschnitts (eine Grafik pro Abschnitt),
    mit Einzelwerten der Probanden und Welch-p-Wert im Titel.
    """

    included = kennwerte[
        ~kennwerte["ausgeschlossen"] & (kennwerte["abschnitt"] == segment)
    ]
    groups = sorted(kennwerte["gruppe"].unique())
    metrics = [
        (m, label) for m, label in kennwert_liste().items()
        if m in included and included[m].notna().any()
    ]

    if not metrics:
        return

    n_cols = 4
    n_rows = int(np.ceil(len(metrics) / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )

    text_color = "#3d3d3a"
    grid_color = "#e4e3dc"
    rng = np.random.default_rng(0)

    for ax, (metric, label) in zip(axes.flat, metrics):

        values = [
            included.loc[included["gruppe"] == g, metric].dropna().to_numpy()
            for g in groups
        ]

        boxes = ax.boxplot(
            values,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": text_color, "linewidth": 2},
            whiskerprops={"color": "#8a8980", "linewidth": 1},
            capprops={"color": "#8a8980", "linewidth": 1},
        )

        for patch, color in zip(boxes["boxes"], GRUPPEN_FARBEN):
            patch.set_facecolor(color + "55")
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)

        for pos, (vals, color) in enumerate(
            zip(values, GRUPPEN_FARBEN), start=1
        ):
            ax.scatter(
                pos + rng.uniform(-0.12, 0.12, len(vals)),
                vals,
                s=16,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

        test = tests[
            (tests["abschnitt"] == segment) & (tests["kennwert"] == metric)
        ] if len(tests) else pd.DataFrame()

        subtitle = (
            f"Welch {format_p(test['welch_p'].iloc[0])}, "
            f"Holm {format_p(test['welch_p_holm'].iloc[0])}"
            if len(test) else ""
        )

        ax.set_title(
            f"{label}\n{subtitle}",
            fontsize=9.5, color=text_color, loc="left",
        )
        ax.set_xticks(
            range(1, len(groups) + 1),
            [GRUPPEN_LABEL.get(g, g) for g in groups],
            fontsize=8, color=text_color,
        )
        ax.tick_params(axis="y", labelsize=8, colors=text_color)
        ax.grid(axis="y", color=grid_color, linewidth=0.8)
        ax.set_axisbelow(True)

        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#b5b4ab")

    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Eyetracking-Kennwerte nach Versuchsgruppe – Abschnitt: {segment}",
        fontsize=13, color=text_color, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_file, dpi=150, facecolor="white")
    plt.close(fig)


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EYETRACKING-AUSWERTUNG – VERGLEICH DER VERSUCHSGRUPPEN")
    print("=" * 70)
    print()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Kennwerte pro Proband
    # --------------------------------------------------------

    if KENNWERTE_NEU_BERECHNEN or not KENNWERTE_FILE.exists():
        print("Kennwerte werden aus den ASC-Dateien berechnet ...")
        kennwerte = compute_kennwerte()
    else:
        print(f"Kennwerte werden geladen: {KENNWERTE_FILE.name}")
        kennwerte = pd.read_csv(KENNWERTE_FILE, encoding="utf-8-sig")
        kennwerte = kennwerte.drop(columns=["ausgeschlossen"])

    # Ausschluss bei schlechter Datenqualität (bezogen auf Gesamtfahrt)
    quality = kennwerte[kennwerte["abschnitt"] == "gesamt"].set_index(
        ["gruppe", "proband"]
    )["anteil_gueltig_prozent"]

    excluded = set(
        quality[quality < 100 * MIN_GUELTIG_ANTEIL].index
    )

    kennwerte["ausgeschlossen"] = [
        (g, p) in excluded
        for g, p in zip(kennwerte["gruppe"], kennwerte["proband"])
    ]

    kennwerte.to_csv(KENNWERTE_FILE, index=False, encoding="utf-8-sig")

    print()
    print("PROBANDEN PRO GRUPPE (eingeschlossen / gesamt):")
    for group, part in kennwerte[kennwerte["abschnitt"] == "gesamt"] \
            .groupby("gruppe"):
        print(
            f"  {group}: {(~part['ausgeschlossen']).sum()} / {len(part)}"
        )

    if excluded:
        print()
        print(
            f"AUSGESCHLOSSEN (< {100 * MIN_GUELTIG_ANTEIL:.0f} % "
            f"gültige Blickdaten):"
        )
        for group, participant_id in sorted(excluded):
            print(
                f"  {group} / {participant_id}: "
                f"{quality[(group, participant_id)]:.1f} %"
            )

    # --------------------------------------------------------
    # Statistik
    # --------------------------------------------------------

    descriptive, tests, posthoc = run_statistics(kennwerte)

    descriptive.to_csv(
        OUTPUT_FOLDER / "deskriptive_statistik.csv",
        index=False, encoding="utf-8-sig",
    )
    tests.to_csv(
        OUTPUT_FOLDER / "anova_ergebnisse.csv",
        index=False, encoding="utf-8-sig",
    )
    posthoc.to_csv(
        OUTPUT_FOLDER / "posthoc_vergleiche.csv",
        index=False, encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Grafiken
    # --------------------------------------------------------

    for segment in ABSCHNITTE:
        plot_boxplots(
            kennwerte, tests, segment,
            OUTPUT_FOLDER / f"boxplots_{segment}.png",
        )

    # --------------------------------------------------------
    # Kurzübersicht in der Konsole
    # --------------------------------------------------------

    print()
    print("WELCH-ANOVA (Holm-korrigiert pro Abschnitt):")

    for segment in ABSCHNITTE:

        print()
        print(f"  Abschnitt: {segment}")

        part = tests[tests["abschnitt"] == segment] if len(tests) else tests

        for _, row in part.iterrows():
            marker = "*" if row["welch_p_holm"] < 0.05 else " "
            print(
                f"   {marker} {row['beschriftung']:<48} "
                f"F({row['welch_df1']:.0f}, {row['welch_df2']:.1f}) = "
                f"{row['welch_F']:6.2f}  {format_p(row['welch_p']):<10} "
                f"Holm {format_p(row['welch_p_holm']):<10} "
                f"η² = {row['eta2']:.2f}"
            )

    print()
    print("=" * 70)
    print("FERTIG")
    print("=" * 70)
    print(f"Ergebnisse: {OUTPUT_FOLDER}")
    print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
