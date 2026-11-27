#!/usr/bin/env python3
import argparse
import csv
import datetime
import glob
import json
import math
import os
import statistics
import sys
import time

ROOT_DIR = os.path.expanduser("~/ws/disaster_recovery_ws/benchmark_results")
OUT_MD = os.path.join(ROOT_DIR, "integrated_15scenarios_comparison.md")
OUT_CSV = os.path.join(ROOT_DIR, "integrated_15scenarios_comparison.csv")

STRATEGIES = [
    ("Baseline",       "Baseline",     "ab_phase4_150",      "baseline"),
    ("Phase 2-B",      "Phase 2-B",    "ab_phase2b_150",     "proposed"),
    ("Phase 3",        "Phase 3",      "ab_phase3_150",      "proposed"),
    ("Phase 4 (Sim)",  "Phase 4 Sim",  "ab_phase4_150",      "proposed"),
    ("Phase 4 (Real)", "Phase 4 Real", "ab_phase4_real_150", "proposed"),
]

C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_DIM    = "\033[2m"
C_CYAN   = "\033[96m"
C_BLUE   = "\033[94m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_RED    = "\033[91m"
C_WHITE  = "\033[97m"


def find_trial_list(obj, depth=0):
    if depth > 3:
        return None
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        if any("success" in x or "arrival_success" in x for x in obj):
            return obj
    if isinstance(obj, dict):
        for k in ("trials", "trial_results", "records", "episodes", "items"):
            if k in obj:
                res = find_trial_list(obj[k], depth + 1)
                if res:
                    return res
        for v in obj.values():
            res = find_trial_list(v, depth + 1)
            if res:
                return res
    return None


def safe_mean(vals):
    clean = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.mean(clean) if clean else None


def extract_scenario_metrics(d, slot):
    if not isinstance(d, dict):
        return None

    candidates = [
        d.get(f"{slot}_trials"),
        d.get(slot),
        d.get(f"{slot}_results"),
        d.get("results", {}).get(f"{slot}_trials") if isinstance(d.get("results"), dict) else None,
        d.get("results", {}).get(slot) if isinstance(d.get("results"), dict) else None,
        d.get("trials", {}).get(slot) if isinstance(d.get("trials"), dict) else None,
    ]
    tlist = None
    for cand in candidates:
        tlist = find_trial_list(cand)
        if tlist:
            break

    if tlist:
        n = len(tlist)
        succ_trials = [t for t in tlist if bool(t.get("success"))]
        succ_cnt = len(succ_trials)
        arr_vals = [bool(t["arrival_success"]) for t in tlist if "arrival_success" in t]
        cf_vals = [bool(t["collision_free"]) for t in tlist if "collision_free" in t]

        return {
            "n": n,
            "succ_cnt": succ_cnt,
            "succ_rate": (succ_cnt / n * 100.0) if n > 0 else None,
            "arr_rate": (sum(arr_vals) / len(arr_vals) * 100.0) if arr_vals else None,
            "cf_rate": (sum(cf_vals) / len(cf_vals) * 100.0) if cf_vals else None,
            "succ_dur_s": safe_mean([t.get("return_duration_s") for t in succ_trials]),
            "min_debris_m": safe_mean([t.get("min_debris_dist_m") for t in tlist]),
            "term_err_cm": safe_mean([t.get("terminal_error_cm") for t in succ_trials]),
        }

    sum_cands = [
        d.get(slot),
        d.get(f"{slot}_summary"),
        d.get("summary", {}).get(slot) if isinstance(d.get("summary"), dict) else None,
        d.get("metrics", {}).get(slot) if isinstance(d.get("metrics"), dict) else None,
    ]
    for sc in sum_cands:
        if not isinstance(sc, dict):
            continue
        n = 10
        for nk in ("n_trials", "total_trials", "num_trials", "trials", "count"):
            if isinstance(sc.get(nk), int) and sc[nk] > 0:
                n = sc[nk]
                break
        succ_cnt, succ_rate = None, None
        for sk in ("success_count", "successes", "n_success", "num_success"):
            if isinstance(sc.get(sk), (int, float)):
                succ_cnt = int(round(sc[sk]))
                succ_rate = succ_cnt / n * 100.0
                break
        if succ_rate is None and isinstance(sc.get("success_rate"), (int, float)):
            sr = float(sc["success_rate"])
            succ_rate = sr * 100.0 if sr <= 1.0 else sr
            succ_cnt = int(round(succ_rate / 100.0 * n))
        if succ_rate is not None:
            return {
                "n": n,
                "succ_cnt": succ_cnt,
                "succ_rate": succ_rate,
                "arr_rate": sc.get("arrival_rate"),
                "cf_rate": sc.get("collision_free_rate"),
                "succ_dur_s": sc.get("mean_return_duration_s") or sc.get("return_duration_s"),
                "min_debris_m": sc.get("mean_min_debris_dist_m") or sc.get("min_debris_dist_m"),
                "term_err_cm": sc.get("mean_terminal_error_cm") or sc.get("terminal_error_cm"),
            }
    return None


def load_directory_scenarios(dname):
    bdir = os.path.join(ROOT_DIR, dname)
    if not os.path.isdir(bdir):
        return {}
    chosen = {}
    for p in sorted(glob.glob(os.path.join(bdir, "ab_*_10trials*.json"))):
        bn = os.path.basename(p)
        if bn.startswith("ab_summary_"):
            continue
        key = bn.replace(".partial.json", ".json")
        if key not in chosen or not bn.endswith(".partial.json"):
            chosen[key] = p
    out = {}
    for key, p in chosen.items():
        scen = key[3:-len("_10trials.json")]
        try:
            st = os.stat(p)
            with open(p, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            birth = getattr(st, "st_birthtime", None) or st.st_ctime
            out[scen] = (birth, p.endswith(".partial.json"), data)
        except Exception:
            continue
    return out


def pad_cell(plain_text, width, align="center", color=""):
    # 반드시 순수 ASCII 평문으로 폭을 먼저 맞춘 뒤 ANSI 색상을 래핑한다.
    txt = str(plain_text)[:width]
    if align == "left":
        padded = txt.ljust(width)
    elif align == "right":
        padded = txt.rjust(width)
    else:
        padded = txt.center(width)
    return f"{color}{padded}{C_RESET}" if color else padded


def rate_color(sr, is_best=False):
    if sr is None:
        return C_DIM
    if is_best and sr > 0:
        return C_BOLD + C_CYAN
    if sr >= 80.0:
        return C_BOLD + C_GREEN
    if sr >= 40.0:
        return C_YELLOW
    return C_RED


def fmt_pct_plain(val, cnt=None, n=None):
    if val is None:
        return "-"
    if cnt is not None and n is not None:
        return f"{val:5.1f}% ({cnt}/{n})"
    return f"{val:5.1f}%"


def fmt_num(val, digits=2):
    return f"{val:.{digits}f}" if val is not None else "-"


def make_table_border(widths, left, mid, right, fill="-"):
    segs = [fill * (w + 2) for w in widths]
    return f"  {C_DIM}{left}{mid.join(segs)}{right}{C_RESET}"


def render_table_row(cells):
    sep = f" {C_DIM}|{C_RESET} "
    return f"  {C_DIM}|{C_RESET} " + sep.join(cells) + f" {C_DIM}|{C_RESET}"


def render_once(watch_mode=False):
    dir_cache = {}
    for _, _, dname, _ in STRATEGIES:
        if dname not in dir_cache:
            dir_cache[dname] = load_directory_scenarios(dname)

    scen_order_map = {}
    for scen, (birth, _, _) in dir_cache.get("ab_phase4_150", {}).items():
        scen_order_map[scen] = (0, birth, scen)
    for _, scens in dir_cache.items():
        for scen in scens:
            if scen not in scen_order_map:
                scen_order_map[scen] = (1, 0.0, scen)
    scenarios = sorted(scen_order_map.keys(), key=lambda s: scen_order_map[s])

    if not scenarios:
        return f"{C_RED}[ERROR] No benchmark JSON files found in {ROOT_DIR}{C_RESET}"

    data = {label: {} for label, _, _, _ in STRATEGIES}
    partial_set = set()

    for label, _, dname, slot in STRATEGIES:
        scens = dir_cache.get(dname, {})
        for scen in scenarios:
            if scen not in scens:
                continue
            _, is_partial, raw_json = scens[scen]
            if is_partial:
                partial_set.add((label, scen))
            m = extract_scenario_metrics(raw_json, slot)
            if m:
                data[label][scen] = m

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # =========================================================================
    # 1. Markdown & CSV 파일 저장
    # =========================================================================
    md_lines = [
        "# Disaster Recovery 15-Scenario Integrated Benchmark Comparison",
        f"- Generated at: `{ts}`",
        f"- Source root: `{ROOT_DIR}`",
    ]
    if partial_set:
        p_str = ", ".join(f"{l}:{s}" for l, s in sorted(partial_set))
        md_lines.append(f"- **[IN-PROGRESS] Partial scenarios included:** `{p_str}`")
    md_lines.append("")
    md_lines.append("## 1. Scenario-wise Success Rate Comparison (15 Scenarios × 5 Strategies)")
    headers_t1 = ["Scenario"] + [s[0] for s in STRATEGIES] + ["Best Strategy"]
    md_lines.append("| " + " | ".join(headers_t1) + " |")
    md_lines.append("| " + " | ".join(["---"] + ["---:"] * len(STRATEGIES) + ["---"]) + " |")

    csv_rows = []
    for scen in scenarios:
        row_cells = [scen]
        best_val = -1.0
        best_names = []
        csv_row = {"scenario": scen}

        for label, _, _, _ in STRATEGIES:
            m = data[label].get(scen)
            if m and m["succ_rate"] is not None:
                sr = m["succ_rate"]
                row_cells.append(fmt_pct_plain(sr, m["succ_cnt"], m["n"]))
                csv_row[f"{label}_succ_rate"] = round(sr, 2)
                csv_row[f"{label}_succ_cnt"] = m["succ_cnt"]
                csv_row[f"{label}_n"] = m["n"]
                csv_row[f"{label}_arr_rate"] = round(m["arr_rate"], 2) if m["arr_rate"] is not None else ""
                csv_row[f"{label}_cf_rate"] = round(m["cf_rate"], 2) if m["cf_rate"] is not None else ""
                csv_row[f"{label}_succ_dur_s"] = round(m["succ_dur_s"], 2) if m["succ_dur_s"] is not None else ""
                csv_row[f"{label}_min_debris_m"] = round(m["min_debris_m"], 3) if m["min_debris_m"] is not None else ""
                csv_row[f"{label}_term_err_cm"] = round(m["term_err_cm"], 2) if m["term_err_cm"] is not None else ""

                if sr > best_val + 1e-6:
                    best_val = sr
                    best_names = [label]
                elif abs(sr - best_val) <= 1e-6 and sr >= 0:
                    best_names.append(label)
            else:
                row_cells.append("-")

        best_str = ", ".join(best_names) if best_names and best_val > 0 else ("All 0%" if best_val == 0 else "-")
        row_cells.append(best_str)
        md_lines.append("| " + " | ".join(row_cells) + " |")
        csv_rows.append(csv_row)

    micro_cells = ["**OVERALL (Micro)**"]
    macro_cells = ["**MACRO AVG (Scen)**"]
    for label, _, _, _ in STRATEGIES:
        ms = [data[label][s] for s in scenarios if s in data[label]]
        tot_n = sum(m["n"] for m in ms)
        tot_s = sum(m["succ_cnt"] for m in ms if m["succ_cnt"] is not None)
        micro_sr = (tot_s / tot_n * 100.0) if tot_n > 0 else None
        macro_sr = safe_mean([m["succ_rate"] for m in ms])
        micro_cells.append(f"**{fmt_pct_plain(micro_sr, tot_s, tot_n)}**" if micro_sr is not None else "-")
        macro_cells.append(f"**{fmt_pct_plain(macro_sr)}**" if macro_sr is not None else "-")
    micro_cells.append("-")
    macro_cells.append("-")
    md_lines.append("| " + " | ".join(micro_cells) + " |")
    md_lines.append("| " + " | ".join(macro_cells) + " |")
    md_lines.append("")

    md_lines.append("## 2. Overall Multi-Metric Summary (150 Trials per Strategy)")
    headers_t2 = [
        "Strategy", "Trials", "Success (%)", "Arrival (%)", "Collision-Free (%)",
        "Succ Time (s)", "Min Debris (m)", "Term Err (cm)", "vs Base (W/T/L)"
    ]
    md_lines.append("| " + " | ".join(headers_t2) + " |")
    md_lines.append("| " + " | ".join(["---"] + ["---:"] * 7 + ["---"]) + " |")

    base_map = data.get("Baseline", {})
    summary_stats = []

    for label, short_lbl, _, _ in STRATEGIES:
        ms = [data[label][s] for s in scenarios if s in data[label]]
        tot_n = sum(m["n"] for m in ms)
        tot_s = sum(m["succ_cnt"] for m in ms if m["succ_cnt"] is not None)
        micro_sr = (tot_s / tot_n * 100.0) if tot_n > 0 else None
        macro_sr = safe_mean([m["succ_rate"] for m in ms])

        def weighted_avg(key):
            num, den = 0.0, 0
            for m in ms:
                if m.get(key) is not None and m["n"] > 0:
                    num += m[key] * m["n"]
                    den += m["n"]
            return (num / den) if den > 0 else None

        def succ_weighted_avg(key):
            num, den = 0.0, 0
            for m in ms:
                sc = m.get("succ_cnt") or 0
                if m.get(key) is not None and sc > 0:
                    num += m[key] * sc
                    den += sc
            return (num / den) if den > 0 else None

        arr_avg = weighted_avg("arr_rate")
        cf_avg = weighted_avg("cf_rate")
        dur_avg = succ_weighted_avg("succ_dur_s")
        deb_avg = weighted_avg("min_debris_m")
        err_avg = succ_weighted_avg("term_err_cm")

        if label == "Baseline":
            wtl_str = "-"
        else:
            w, t, l = 0, 0, 0
            for s in scenarios:
                bm = base_map.get(s)
                pm = data[label].get(s)
                if bm and pm and bm["succ_rate"] is not None and pm["succ_rate"] is not None:
                    if pm["succ_rate"] > bm["succ_rate"] + 1e-6:
                        w += 1
                    elif pm["succ_rate"] < bm["succ_rate"] - 1e-6:
                        l += 1
                    else:
                        t += 1
            wtl_str = f"{w}W / {t}T / {l}L"

        summary_stats.append({
            "label": label,
            "short": short_lbl,
            "tot_n": tot_n,
            "tot_s": tot_s,
            "micro_sr": micro_sr,
            "macro_sr": macro_sr,
            "arr_avg": arr_avg,
            "cf_avg": cf_avg,
            "dur_avg": dur_avg,
            "deb_avg": deb_avg,
            "err_avg": err_avg,
            "wtl_str": wtl_str,
        })

        row_t2 = [
            f"**{label}**",
            f"{tot_n}/150",
            fmt_pct_plain(micro_sr, tot_s, tot_n),
            fmt_pct_plain(arr_avg),
            fmt_pct_plain(cf_avg),
            fmt_num(dur_avg, 2),
            fmt_num(deb_avg, 3),
            fmt_num(err_avg, 2),
            wtl_str,
        ]
        md_lines.append("| " + " | ".join(row_t2) + " |")

    with open(OUT_MD, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md_lines) + "\n")

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        for r in csv_rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    # =========================================================================
    # 2. 순수 반각 폭 보장 터미널 테이블 렌더링 (총 폭 116열)
    # =========================================================================
    W1 = [20, 13, 13, 13, 13, 13, 17]  # sum=102 + 22(구분선) = 124 -> 여백 포함 126열
    W1 = [19, 13, 13, 13, 13, 13, 15]  # 총 폭 = 2 + 99 + 21 = 122 -> 압축:
    W1 = [18, 12, 12, 12, 12, 12, 14]  # 총 폭 = 2 + 92 + 21 = 115열

    mode_str = "[LIVE WATCH - Ctrl+C]" if watch_mode else "[SNAPSHOT REPORT]"
    title_plain = f"DISASTER RECOVERY BENCHMARK MATRIX   |   {ts}   |   {mode_str}"
    box_w = 113

    tui = [
        f"  {C_BOLD}{C_BLUE}+{'=' * (box_w - 2)}+{C_RESET}",
        f"  {C_BOLD}{C_BLUE}|{C_RESET} {pad_cell(title_plain, box_w - 4, 'left', C_BOLD + C_WHITE)} {C_BOLD}{C_BLUE}|{C_RESET}",
        f"  {C_BOLD}{C_BLUE}+{'=' * (box_w - 2)}+{C_RESET}",
        "",
        f"  {C_BOLD}{C_CYAN}[1] SCENARIO-WISE SUCCESS MATRIX{C_RESET}  {C_DIM}(> = in-progress partial | * = best strategy){C_RESET}",
        make_table_border(W1, "+", "+", "+", "-"),
        render_table_row(
            [pad_cell("SCENARIO", W1[0], "left", C_BOLD)]
            + [pad_cell(s[1], W1[i + 1], "center", C_BOLD) for i, s in enumerate(STRATEGIES)]
            + [pad_cell("LEADER", W1[6], "left", C_BOLD)]
        ),
        make_table_border(W1, "+", "+", "+", "="),
    ]

    for scen in scenarios:
        is_any_partial = any((lbl, scen) in partial_set for lbl, _, _, _ in STRATEGIES)
        scen_plain = f"> {scen}" if is_any_partial else f"  {scen}"
        scen_cell = pad_cell(scen_plain, W1[0], "left", C_YELLOW if is_any_partial else "")

        best_val = -1.0
        best_short = []
        for label, short_lbl, _, _ in STRATEGIES:
            m = data[label].get(scen)
            if m and m["succ_rate"] is not None:
                sr = m["succ_rate"]
                if sr > best_val + 1e-6:
                    best_val = sr
                    best_short = [short_lbl]
                elif abs(sr - best_val) <= 1e-6 and sr >= 0:
                    best_short.append(short_lbl)

        cells = [scen_cell]
        for i, (label, short_lbl, _, _) in enumerate(STRATEGIES):
            m = data[label].get(scen)
            if m and m["succ_rate"] is not None:
                sr = m["succ_rate"]
                raw_txt = f"{sr:4.0f}%({m['succ_cnt']:2d}/{m['n']:2d})" if sr == 100.0 else f"{sr:4.1f}%({m['succ_cnt']:2d}/{m['n']:2d})"
                is_best = best_val > 0 and (short_lbl in best_short)
                cells.append(pad_cell(raw_txt, W1[i + 1], "center", rate_color(sr, is_best)))
            else:
                cells.append(pad_cell("-", W1[i + 1], "center", C_DIM))

        if best_val > 0:
            leader_plain = "* " + "/".join(best_short)
            cells.append(pad_cell(leader_plain, W1[6], "left", C_BOLD + C_GREEN))
        elif best_val == 0:
            cells.append(pad_cell("All 0.0%", W1[6], "left", C_DIM))
        else:
            cells.append(pad_cell("-", W1[6], "left", C_DIM))

        tui.append(render_table_row(cells))

    tui.append(make_table_border(W1, "+", "+", "+", "-"))

    micro_cells_tui = [pad_cell("OVERALL (Micro)", W1[0], "left", C_BOLD)]
    macro_cells_tui = [pad_cell("MACRO AVG (Scen)", W1[0], "left", C_BOLD)]
    for i, st in enumerate(summary_stats):
        if st["micro_sr"] is not None:
            m_txt = f"{st['micro_sr']:4.1f}%({st['tot_s']}/{st['tot_n']})"
            micro_cells_tui.append(pad_cell(m_txt, W1[i + 1], "center", rate_color(st["micro_sr"], False)))
        else:
            micro_cells_tui.append(pad_cell("-", W1[i + 1], "center", C_DIM))

        if st["macro_sr"] is not None:
            ma_txt = f"{st['macro_sr']:5.1f}%"
            macro_cells_tui.append(pad_cell(ma_txt, W1[i + 1], "center", rate_color(st["macro_sr"], False)))
        else:
            macro_cells_tui.append(pad_cell("-", W1[i + 1], "center", C_DIM))

    micro_cells_tui.append(pad_cell("-", W1[6], "left", C_DIM))
    macro_cells_tui.append(pad_cell("-", W1[6], "left", C_DIM))
    tui.append(render_table_row(micro_cells_tui))
    tui.append(render_table_row(macro_cells_tui))
    tui.append(make_table_border(W1, "+", "+", "+", "-"))

    # =========================================================================
    # 3. Table 2: 다중 평가지표 요약 카드 (총 폭 동일하게 113열 맞춤)
    # =========================================================================
    W2 = [14, 19, 13, 7, 7, 7, 8, 7, 11]  # sum=93 + 28(구분선) = 121 -> 113열 맞춤:
    W2 = [14, 17, 13, 6, 7, 7, 7, 6, 10]  # sum=87 + 28 = 115 -> 113열:
    W2 = [14, 16, 12, 6, 7, 7, 7, 6, 10]  # sum=85 + 28 = 113열 정확 일치

    tui.extend([
        "",
        f"  {C_BOLD}{C_CYAN}[2] MULTI-METRIC PERFORMANCE SUMMARY{C_RESET}  {C_DIM}(TIME & ERR = succ trials | DEBRIS = all trials){C_RESET}",
        make_table_border(W2, "+", "+", "+", "-"),
        render_table_row([
            pad_cell("STRATEGY", W2[0], "left", C_BOLD),
            pad_cell("PROGRESS", W2[1], "center", C_BOLD),
            pad_cell("SUCCESS", W2[2], "center", C_BOLD),
            pad_cell("ARRIVE", W2[3], "right", C_BOLD),
            pad_cell("NO-COLL", W2[4], "right", C_BOLD),
            pad_cell("TIME(s)", W2[5], "right", C_BOLD),
            pad_cell("DEBRIS", W2[6], "right", C_BOLD),
            pad_cell("ERR", W2[7], "right", C_BOLD),
            pad_cell("vs BASE", W2[8], "center", C_BOLD),
        ]),
        make_table_border(W2, "+", "+", "+", "="),
    ])

    for st in summary_stats:
        bar_w = 8
        ratio = min(1.0, max(0.0, st["tot_n"] / 150.0))
        filled = int(round(bar_w * ratio))
        prog_plain = f"[{'#' * filled}{'-' * (bar_w - filled)}] {st['tot_n']:3d}"

        if st["micro_sr"] is not None:
            s_txt = f"{st['micro_sr']:4.1f}%({st['tot_s']})"
            succ_cell = pad_cell(s_txt, W2[2], "center", rate_color(st["micro_sr"], False))
        else:
            succ_cell = pad_cell("-", W2[2], "center", C_DIM)

        arr_txt = f"{st['arr_avg']:5.1f}%" if st["arr_avg"] is not None else "-"
        cf_txt  = f"{st['cf_avg']:5.1f}%" if st["cf_avg"] is not None else "-"
        dur_txt = f"{st['dur_avg']:6.1f}" if st["dur_avg"] is not None else "-"
        deb_txt = f"{st['deb_avg']:6.3f}" if st["deb_avg"] is not None else "-"
        err_txt = f"{st['err_avg']:5.1f}" if st["err_avg"] is not None else "-"
        wtl_col = C_GREEN if st["wtl_str"] != "-" else C_DIM

        tui.append(render_table_row([
            pad_cell(st["label"], W2[0], "left", C_BOLD),
            pad_cell(prog_plain, W2[1], "center", C_GREEN if st["tot_n"] > 0 else C_DIM),
            succ_cell,
            pad_cell(arr_txt, W2[3], "right"),
            pad_cell(cf_txt, W2[4], "right"),
            pad_cell(dur_txt, W2[5], "right"),
            pad_cell(deb_txt, W2[6], "right"),
            pad_cell(err_txt, W2[7], "right"),
            pad_cell(st["wtl_str"], W2[8], "center", wtl_col),
        ]))

    tui.extend([
        make_table_border(W2, "+", "+", "+", "-"),
        f"  {C_DIM}Saved -> MD: {OUT_MD}  |  CSV: {OUT_CSV}{C_RESET}",
    ])
    return "\n".join(tui)


def main():
    parser = argparse.ArgumentParser(description="15-Scenario Multi-Strategy Benchmark Matrix Dashboard")
    parser.add_argument("--watch", action="store_true", help="Refresh live dashboard continuously without screen flicker")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds (default: 5.0)")
    args = parser.parse_args()

    if not args.watch:
        print(render_once(watch_mode=False))
        return

    try:
        sys.stdout.write("\033[2J")
        while True:
            screen = render_once(watch_mode=True)
            sys.stdout.write("\033[H" + screen + "\033[J\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}[DASHBOARD] 실시간 비교표 대시보드를 종료했다. 백그라운드 벤치마크는 계속 실행된다.{C_RESET}")


if __name__ == "__main__":
    main()
