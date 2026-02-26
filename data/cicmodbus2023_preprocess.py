# data/cicmodbus2023_preprocess.py
from __future__ import annotations

from pathlib import Path
import subprocess
import pandas as pd
import shutil
import os



def run_zeek(pcap_path: Path, out_dir: Path) -> Path:
    """
    Run Zeek (the network analyzer) on a single PCAP.
    Returns path to conn.log.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to find the correct Zeek binary.
    # Users sometimes have a different "zeek" command installed (e.g., a Python package CLI).
    zeek_bin = os.environ.get("ZEEK_BIN") or shutil.which("zeek")

    if not zeek_bin:
        raise FileNotFoundError(
            "Could not find Zeek binary. Install via Homebrew: brew install zeek\n"
            "or set environment variable ZEEK_BIN=/full/path/to/zeek"
        )

    # Run: zeek -r <pcap>

    try:
        subprocess.run([zeek_bin, "-C", "-r", str(pcap_path)], cwd=str(out_dir), check=True)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Zeek failed. This often happens when the 'zeek' command is NOT the Zeek network analyzer.\n"
            f"Resolved zeek_bin={zeek_bin}\n"
            f"Try running in terminal:\n"
            f"  {zeek_bin} -r '{pcap_path}'\n"
            f"If that shows 'No such option: -r', you are calling the wrong zeek.\n"
            f"Fix by setting ZEEK_BIN to your Homebrew zeek path (e.g. /opt/homebrew/bin/zeek)."
        ) from e

    conn_log = out_dir / "conn.log"
    if not conn_log.exists():
        raise FileNotFoundError(f"Zeek did not produce conn.log for {pcap_path}")
    return conn_log


def read_conn_log(conn_log: Path) -> pd.DataFrame:
    """
    Read Zeek conn.log using the '#fields' header to get correct column names.
    Zeek logs are tab-separated, with metadata lines starting with '#'.
    """
    fields = None
    with conn_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#fields"):
                # Example: #fields ts uid id.orig_h id.orig_p id.resp_h ...
                fields = line.strip().split("\t")[1:]
                break

    if not fields:
        raise ValueError(f"Could not find '#fields' header in {conn_log}")

    df = pd.read_csv(
        conn_log,
        sep="\t",
        comment="#",
        header=None,
        names=fields,
        low_memory=False
    )
    return df


def preprocess(raw_dir: Path,
               out_csv: Path,
               tmp_dir: Path,
               max_benign: int | None = None,
               max_attack: int | None = None) -> None:
    """
    Convert CIC Modbus Dataset 2023 PCAPs into a labeled flow dataset.

    Labeling strategy (simple & robust):
    - any flow extracted from benign/ PCAPs => label=0
    - any flow extracted from attack/  PCAPs => label=1
    """
    benign_pcaps = sorted((raw_dir / "benign").rglob("*.pcap"))
    attack_pcaps = sorted((raw_dir / "attack").rglob("*.pcap"))

    if max_benign is not None:
        benign_pcaps = benign_pcaps[:max_benign]
    if max_attack is not None:
        attack_pcaps = attack_pcaps[:max_attack]

    print(f"Found benign pcaps: {len(benign_pcaps)}")
    print(f"Found attack  pcaps: {len(attack_pcaps)}")

    all_rows = []

    def process_one(pcap_path: Path, label: int):
        # Each pcap gets its own Zeek output folder so logs never overwrite
        rel = pcap_path.relative_to(raw_dir)
        zeek_out = tmp_dir / rel.parent / pcap_path.stem

        conn_log = run_zeek(pcap_path, zeek_out)
        df = read_conn_log(conn_log)

        # Add dataset metadata (useful later for analysis)
        df["label"] = label
        df["pcap_file"] = pcap_path.name
        df["pcap_relpath"] = str(rel)

        return df

    # --- Process benign ---
    for i, pcap in enumerate(benign_pcaps, start=1):
        print(f"[benign] {i}/{len(benign_pcaps)} {pcap.name}", flush=True)
        all_rows.append(process_one(pcap, label=0))

    # --- Process attack ---
    for i, pcap in enumerate(attack_pcaps, start=1):
        print(f"[attack] {i}/{len(attack_pcaps)} {pcap.name}", flush=True)
        all_rows.append(process_one(pcap, label=1))

    flows = pd.concat(all_rows, ignore_index=True)
    rename_map = {
        "id.orig_h": "src_ip",
        "id.orig_p": "src_port",
        "id.resp_h": "dst_ip",
        "id.resp_p": "dst_port",
    }
    flows = flows.rename(columns={k: v for k, v in rename_map.items() if k in flows.columns})

    # Convert numeric columns (Zeek sometimes uses '-' for missing)
    for c in ["id.orig_p", "id.resp_p", "duration", "orig_bytes", "resp_bytes"]:
        if c in flows.columns:
            flows[c] = pd.to_numeric(flows[c], errors="coerce").fillna(0.0)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    flows.to_csv(out_csv, index=False)
    print(f"Saved labeled flows: {out_csv}")


def main():

    # python -m data.cicmodbus2023_preprocess --test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="datasets/CICModbus2023/raw")
    parser.add_argument("--out", type=str, default="datasets/CICModbus2023/processed/flows_labeled.csv")
    parser.add_argument("--tmp", type=str, default="datasets/CICModbus2023/tmp_zeek")
    parser.add_argument("--test", action="store_true", help="Run a tiny preprocessing test (3 benign + 3 attack)")
    args = parser.parse_args()

    raw_dir = Path(args.raw).resolve()
    out_csv = Path(args.out).resolve()
    tmp_dir = Path(args.tmp).resolve()

    if args.test:
        preprocess(raw_dir, out_csv, tmp_dir, max_benign=3, max_attack=3)
    else:
        preprocess(raw_dir, out_csv, tmp_dir, max_benign=None, max_attack=None)


if __name__ == "__main__":
    main()
