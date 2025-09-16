import argparse, os, random
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to counsel_chat.csv")
    ap.add_argument("--outdir", default="mhbench/raw/counselchat_split", help="Output folder")
    ap.add_argument("--test-frac", type=float, default=0.10, help="Fraction of questions for TEST")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load the CSV
    df = pd.read_csv(args.csv)

    # Drop rows missing questionID or answerText
    df = df.dropna(subset=["questionID", "answerText"])
    df["questionID"] = df["questionID"].astype(str).str.strip()

    # Unique question IDs
    qids = sorted(df["questionID"].unique().tolist())

    # Shuffle and split deterministically
    random.seed(args.seed)
    random.shuffle(qids)
    k_test = max(1, int(len(qids) * args.test_frac))
    test_qids = set(qids[:k_test])
    train_qids = set(qids[k_test:])

    # Split the dataframe
    df_train = df[df["questionID"].isin(train_qids)].copy()
    df_test = df[df["questionID"].isin(test_qids)].copy()

    # Save splits
    train_csv = os.path.join(args.outdir, "counselchat_train.csv")
    test_csv  = os.path.join(args.outdir, "counselchat_test.csv")
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    # Save IDs for reproducibility
    with open(os.path.join(args.outdir, "train_ids.txt"), "w", encoding="utf-8") as f:
        for q in sorted(train_qids):
            f.write(q + "\n")
    with open(os.path.join(args.outdir, "test_ids.txt"), "w", encoding="utf-8") as f:
        for q in sorted(test_qids):
            f.write(q + "\n")

    # Print stats
    print("=== CounselChat Split Summary ===")
    print(f"Total: {len(df)} rows, {df['questionID'].nunique()} unique questions")
    print(f"Train: {len(df_train)} rows, {df_train['questionID'].nunique()} unique questions")
    print(f"Test:  {len(df_test)} rows, {df_test['questionID'].nunique()} unique questions")
    print(f"Files written:\n - {train_csv}\n - {test_csv}")

if __name__ == "__main__":
    main()
