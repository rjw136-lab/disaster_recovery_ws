#!/usr/bin/env python3
import argparse
import csv
import datetime
import glob
import json
import math
import os
import sys
import time

ROOT_DIR = os.path.expanduser("~/ws/disaster_recovery_ws/benchmark_results")
OUT_MD = os.path.join(ROOT_DIR, "failure_breakdown_5strategies.md")
OUT_CSV = os.path.join(ROOT_DIR, "failure_breakdown_5strategies.csv")

STRATEGIES = [
    ("Baseline",       "ab_phase4_150",      "baseline"),
    ("Phase 2-B",      "ab_phase2b_150",     "proposed"),
    ("Phase 3",        "ab_phase3_150",      "proposed"),
    ("Phase 4 (Sim)",  "ab_phase4_150",      "proposed"),
    ("Phase 4 (Real)", "ab_phase4_real_150", "proposed"),
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


def extract_trials(d, slot):
    if not isinstance(d, dict):
        return []
    candidates = [
        d.get(f"{slot}_trials"),
        d.get(slot),
        d.get(f"{slot}_results"),
        d.get("results", {}).get(f"{slot}_trials") if isinstance(d.get("results"), dict) else None,
        d.get("results", {}).get(slot) if isinstance(d.get("results"), dict) else None,
        d.get("trials", {}).get(slot) if isinstance(d.get("trials"), dict) else None,
    ]
    for cand in candidates:
        tlist = find_trial_list(cand)
        if tlist:
            return tlist
    return []


def classify_failure(t, debris_thresh_m=0.18):
    if t.get("debris_collision") is True:
        return "debris_collision"
    if t.get("lidar_collision") is True:
        return "lidar_collision"
    if t.get("timeout") is True or t.get("timed_out") is True:
        return "timeout"

    reason_blob = " ".join(
        str(t.get(k, ""))
        for k in (
            "fail_reason",
            "failure_reason",
            "reason",
            "termination_reason",
            "collision_type",
            "status",
            "final_ctrl_state",
        )
    ).lower()

    if "debris" in reason_blob:
        return "debris_collision"
    if "lidar" in reason_blob or "wall" in reason_blob or "obstacle" in reason_blob or "scan" in reason_blob:
        return "lidar_collision"
    if "timeout" in reason_blob or "time_out" in reason_blob or "max_time" in reason_blob or "expired" in reason_blob:
        return "timeout"

    cf = t.get("collision_free")
    arr = t.get("arrival_success")
    mdeb = t.get("min_debris_dist_m")

    if cf is False:
        if isinstance(mdeb, (int, float)) and math.isfinite(float(mdeb)) and float(mdeb) <= debris_thresh_m:
            return "debris_collision"
        return "lidar_collision"

    if cf is True and arr is False:
        return "timeout"

    if arr is False:
        return "timeout"

    return "other_fail"


def load_directory_scenarios(dname):
    bdir = os.path.join(ROOT_DIR, dname)
    if not os.path.isdir(bdir):
        return {}
    chosen = {}
    for p in sorted(glob.glob(os.path.join(bdir, "ab_*_10trials*.json"))):
        bn = os.path.basename(p)
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
    txt = str(plain_text)[:width]
    if align == "left":
        padded = txt.ljust(width)
    elif align == "right":
        padded = txt.rjust(width)
    else:
        padded = txt.center(width)
    return f"{color}{padded}{C_RESET}" if color else padded


def make_border(widths, fill="-"):
    return "  " + C_DIM + "+" + "+".join(fill * (w + 2) for w in widths) + "+" + C_RESET


def render_row(cells):
    sep = f" {C_DIM}|{C_RESET} "
    return f"  {C_DIM}|{C_RESET} " + sep.join(cells) + f" {C_DIM}|{C_RESET}"


def make_ratio_bar(lidar_c, debris_c, time_c, other_c, width=14):
    tot = lidar_c + debris_c + time_c + other_c
    if tot <= 0:
        return C_DIM + ("-" * width) + C_RESET
    l_w = int(round(width * lidar_c / tot))
    d_w = int(round(width * debris_c / tot))
    t_w = int(round(width * time_c / tot))
    o_w = max(0, width - (l_w + d_w + t_w))
    if l_w + d_w + t_w > width:
        t_w = max(0, width - l_w - d_w)
        if l_w + d_w > width:
            d_w = max(0, width - l_w)
    return (
        f"{C_RED}{'L' * l_w}"
        f"{C_YELLOW}{'D' * d_w}"
        f"{C_CYAN}{'T' * t_w}"
        f"{C_DIM}{'.' * o_w}{C_RESET}"
    )


def render_once(debris_thresh=0.18, show_detail=True, watch_mode=False):
    dir_cache = {}
    for _, dname, _ in STRATEGIES:
        if dname not in dir_cache:
            dir_cache[dname] = load_directory_scenarios(dname)

    scen_order = {}
    for scen, (birth, _, _) in dir_cache.get("ab_phase4_150", {}).items():
        scen_order[scen] = (0, birth, scen)
    for _, scens in dir_cache.items():
        for scen in scens:
            if scen not in scen_order:
                scen_order[scen] = (1, 0.0, scen)
    scenarios = sorted(scen_order.keys(), key=lambda s: scen_order[s])

    if not scenarios:
        return f"{C_RED}[ERROR] No benchmark JSON files found in {ROOT_DIR}{C_RESET}"

    summary_rows = []
    detail_rows = []

    for label, dname, slot in STRATEGIES:
        scens = dir_cache.get(dname, {})
        tot_n = 0
        succ_n = 0
        counts = {
            "lidar_collision": 0,
            "debris_collision": 0,
            "timeout": 0,
            "other_fail": 0,
        }

        for scen in scenarios:
            if scen not in scens:
                continue
            _, is_partial, raw_json = scens[scen]
            tlist = extract_trials(raw_json, slot)
            s_counts = {"lidar_collision": 0, "debris_collision": 0, "timeout": 0, "other_fail": 0}
            s_succ = 0

            for t in tlist:
                tot_n += 1
                if bool(t.get("success")):
                    succ_n += 1
                    s_succ += 1
                else:
                    cat = classify_failure(t, debris_thresh_m=debris_thresh)
                    counts[cat] += 1
                    s_counts[cat] += 1

            if tlist:
                fail_n = len(tlist) - s_succ
                detail_rows.append({
                    "strategy": label,
                    "scenario": scen,
                    "is_partial": is_partial,
                    "trials": len(tlist),
                    "succ": s_succ,
                    "fail": fail_n,
                    "lidar_collision": s_counts["lidar_collision"],
                    "debris_collision": s_counts["debris_collision"],
                    "timeout": s_counts["timeout"],
                    "other_fail": s_counts["other_fail"],
                })

        fail_tot = tot_n - succ_n
        summary_rows.append({
            "strategy": label,
            "trials": tot_n,
            "succ": succ_n,
            "fail": fail_tot,
            **counts,
        })

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    box_w = 113
    mode_str = "[LIVE WATCH - Ctrl+C]" if watch_mode else "[SNAPSHOT REPORT]"
    title_plain = f"5-STRATEGY FAILURE MODE ANALYZER   |   {ts}   |   {mode_str}"

    # Table 1 열 폭 합계: sum=85 + 28(구분선/패딩) = 113열
    W1 = [14, 7, 5, 5, 14, 14, 13, 5, 14]

    lines = [
        f"  {C_BOLD}{C_BLUE}+{'=' * (box_w - 2)}+{C_RESET}",
        f"  {C_BOLD}{C_BLUE}|{C_RESET} {pad_cell(title_plain, box_w - 4, 'left', C_BOLD + C_WHITE)} {C_BOLD}{C_BLUE}|{C_RESET}",
        f"  {C_BOLD}{C_BLUE}+{'=' * (box_w - 2)}+{C_RESET}",
        "",
        f"  {C_BOLD}{C_CYAN}[1] STRATEGY FAILURE SUMMARY{C_RESET}  {C_DIM}(cell: count [all%/fail%] | bar: L=Lidar, D=Debris, T=Timeout){C_RESET}",
        make_border(W1, "-"),
        render_row([
            pad_cell("STRATEGY", W1[0], "left", C_BOLD),
            pad_cell("TRIALS", W1[1], "right", C_BOLD),
            pad_cell("SUCC", W1[2], "right", C_BOLD),
            pad_cell("FAIL", W1[3], "right", C_BOLD),
            pad_cell("LIDAR_COLL", W1[4], "center", C_BOLD + C_RED),
            pad_cell("DEBRIS_COLL", W1[5], "center", C_BOLD + C_YELLOW),
            pad_cell("TIMEOUT", W1[6], "center", C_BOLD + C_CYAN),
            pad_cell("OTHER", W1[7], "right", C_BOLD),
            pad_cell("FAIL MIX", W1[8], "center", C_BOLD),
        ]),
        make_border(W1, "="),
    ]

    for r in summary_rows:
        n = r["trials"]
        f_cnt = r["fail"]

        def fmt_cell(val):
            if n == 0:
                return "-"
            pct_all = val / n * 100.0
            pct_fail = (val / f_cnt * 100.0) if f_cnt > 0 else 0.0
            return f"{val:2d}({pct_all:4.0f}%/{pct_fail:3.0f}%)"

        ratio_bar = make_ratio_bar(
            r["lidar_collision"], r["debris_collision"], r["timeout"], r["other_fail"], width=W1[8]
        )

        lines.append(render_row([
            pad_cell(r["strategy"], W1[0], "left", C_BOLD),
            pad_cell(f"{n}/150" if n > 0 else "-", W1[1], "right"),
            pad_cell(str(r["succ"]) if n > 0 else "-", W1[2], "right", C_GREEN if r["succ"] > 0 else C_DIM),
            pad_cell(str(f_cnt) if n > 0 else "-", W1[3], "right", C_RED if f_cnt > 0 else C_DIM),
            pad_cell(fmt_cell(r["lidar_collision"]), W1[4], "center", C_RED if r["lidar_collision"] > 0 else C_DIM),
            pad_cell(fmt_cell(r["debris_collision"]), W1[5], "center", C_YELLOW if r["debris_collision"] > 0 else C_DIM),
            pad_cell(fmt_cell(r["timeout"]), W1[6], "center", C_CYAN if r["timeout"] > 0 else C_DIM),
            pad_cell(str(r["other_fail"]) if n > 0 else "-", W1[7], "right", C_DIM),
            ratio_bar,
        ]))

    lines.append(make_border(W1, "-"))

    if show_detail:
        # Table 2 열 폭 합계: sum=85 + 28(구분선/패딩) = 113열 정확 일치
        Wd = [14, 19, 6, 5, 5, 11, 11, 9, 5]
        lines.extend([
            "",
            f"  {C_BOLD}{C_CYAN}[2] SCENARIO-LEVEL FAILURE DETAIL{C_RESET}  {C_DIM}(> = in-progress partial | debris thresh <= {debris_thresh:.2f}m){C_RESET}",
            make_border(Wd, "-"),
            render_row([
                pad_cell("STRATEGY", Wd[0], "left", C_BOLD),
                pad_cell("SCENARIO", Wd[1], "left", C_BOLD),
                pad_cell("TRIAL", Wd[2], "right", C_BOLD),
                pad_cell("SUCC", Wd[3], "right", C_BOLD),
                pad_cell("FAIL", Wd[4], "right", C_BOLD),
                pad_cell("LIDAR_COLL", Wd[5], "right", C_BOLD + C_RED),
                pad_cell("DEBRIS_COLL", Wd[6], "right", C_BOLD + C_YELLOW),
                pad_cell("TIMEOUT", Wd[7], "right", C_BOLD + C_CYAN),
                pad_cell("OTHER", Wd[8], "right", C_BOLD),
            ]),
            make_border(Wd, "="),
        ])

        has_any_detail = False
        for dr in detail_rows:
            if dr["fail"] == 0 and not dr["is_partial"]:
                continue
            has_any_detail = True
            scen_txt = f"> {dr['scenario']}" if dr["is_partial"] else f"  {dr['scenario']}"
            scen_col = C_YELLOW if dr["is_partial"] else ""
            lines.append(render_row([
                pad_cell(dr["strategy"], Wd[0], "left"),
                pad_cell(scen_txt, Wd[1], "left", scen_col),
                pad_cell(f"{dr['trials']}/10", Wd[2], "right"),
                pad_cell(str(dr["succ"]), Wd[3], "right", C_GREEN if dr["succ"] > 0 else C_DIM),
                pad_cell(str(dr["fail"]), Wd[4], "right", C_RED if dr["fail"] > 0 else C_DIM),
                pad_cell(str(dr["lidar_collision"]), Wd[5], "right", C_RED if dr["lidar_collision"] > 0 else C_DIM),
                pad_cell(str(dr["debris_collision"]), Wd[6], "right", C_YELLOW if dr["debris_collision"] > 0 else C_DIM),
                pad_cell(str(dr["timeout"]), Wd[7], "right", C_CYAN if dr["timeout"] > 0 else C_DIM),
                pad_cell(str(dr["other_fail"]), Wd[8], "right", C_DIM),
            ]))

        if not has_any_detail:
            lines.append(render_row([
                pad_cell("No failed trials recorded yet.", 109, "center", C_DIM)
            ]))
        lines.append(make_border(Wd, "-"))

    # Markdown & CSV 파일 저장
    md_out = [
        "# 5-Strategy Failure Mode Breakdown",
        f"- Generated at: `{ts}`",
        "",
        "| Strategy | Trials | Success | Fail | lidar_collision | debris_collision | timeout | other_fail |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary_rows:
        n, f_cnt = r["trials"], r["fail"]
        def md_cell(v):
            if n == 0:
                return "-"
            return f"{v} ({v/n*100:.1f}% / {(v/f_cnt*100 if f_cnt else 0):.1f}%)"
        md_out.append(
            f"| **{r['strategy']}** | {n}/150 | {r['succ']} | {r['fail']} | "
            f"{md_cell(r['lidar_collision'])} | {md_cell(r['debris_collision'])} | {md_cell(r['timeout'])} | {r['other_fail']} |"
        )

    with open(OUT_MD, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md_out) + "\n")

    if detail_rows:
        csv_fields = ["strategy", "scenario", "trials", "succ", "fail", "lidar_collision", "debris_collision", "timeout", "other_fail"]
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.DictWriter(fp, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(detail_rows)

    lines.append(f"  {C_DIM}Saved -> MD: {OUT_MD}  |  CSV: {OUT_CSV}{C_RESET}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Real-time 5-Strategy Failure Mode Breakdown Dashboard")
    parser.add_argument("--watch", action="store_true", help="Refresh live dashboard continuously without screen flicker")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds (default: 5.0)")
    parser.add_argument("--debris-thresh", type=float, default=0.18, help="Fallback min_debris_dist_m threshold in meters (default: 0.18)")
    parser.add_argument("--no-detail", action="store_true", help="Hide scenario-level detail table")
    args = parser.parse_args()

    if not args.watch:
        print(render_once(debris_thresh=args.debris_thresh, show_detail=not args.no_detail, watch_mode=False))
        return

    try:
        sys.stdout.write("\033[2J")
        while True:
            screen = render_once(debris_thresh=args.debris_thresh, show_detail=not args.no_detail, watch_mode=True)
            sys.stdout.write("\033[H" + screen + "\033[J\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}[DASHBOARD] 실시간 실패 원인 분석기를 종료했다. 백그라운드 벤치마크는 계속 실행된다.{C_RESET}")


if __name__ == "__main__":
    main()
