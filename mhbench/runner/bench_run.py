import argparse, os, json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from utils.common import read_jsonl, write_jsonl, strip_html, normspace, ensure_dir
from runner.model_hf import HFCausalWrapper
from runner.prompts import *
from metrics.classification import compute_cls_metrics, expected_calibration_error
from metrics.similarity import bertscore_max_multi_ref
from metrics.toxicity import ToxicityScorer
from metrics.fluency import PerplexityScorer


# ------------------------
# IMHI runner
# ------------------------
def build_label_set_from_targets(records):
    labels = sorted({ normspace(r["target"].get("label","")) for r in records if r.get("target") })
    return [l for l in labels if l]

def run_imhi(records, model, outdir):
    label_set = build_label_set_from_targets(records)
    hyps, trues, confs, pred_rows = [], [], [], []

    for r in tqdm(records, desc="IMHI"):
        text = r["input"]["text"]
        labels = r.get("choices") or label_set
        prompt = (IMHI_CHOICE_PROMPT if r.get("choices") else IMHI_OPEN_PROMPT).format(
            labels=", ".join(labels), input=text
        )
        ans = model.classify_json(prompt)
        pred = normspace(ans.get("label",""))
        conf = ans.get("confidence", None)
        true = normspace(r["target"]["label"])

        hyps.append(pred); trues.append(true); confs.append(conf)
        pred_rows.append({
            "id": r["id"], "input": text, "pred": pred, "true": true,
            "confidence": conf, "raw": ans.get("raw")
        })

    # Metrics
    labels_sorted = sorted(set(trues) | set(hyps))
    cls = compute_cls_metrics(trues, hyps, labels=labels_sorted)
    cal = expected_calibration_error(trues, hyps, confs, bins=10)

    ensure_dir(outdir)
    pd.DataFrame(pred_rows).to_csv(os.path.join(outdir, "imhi_predictions.csv"), index=False)
    with open(os.path.join(outdir, "imhi_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"classification": cls, "calibration": cal}, f, indent=2)

    return {"classification": cls, "calibration": cal}


# ------------------------
# ESConv runner
# ------------------------
def run_esconv(records, model, outdir, tox: ToxicityScorer, ref_ppl: PerplexityScorer):
    hyps, refs, pred_rows = [], [], []

    for r in tqdm(records, desc="ESConv"):
        seeker = r["input"]["messages"][0]["text"]
        prompt = DIALOGUE_PROMPT.format(seeker=seeker)
        gen = model.generate(prompt)
        gold = strip_html(r["target"]["reply"])
        hyps.append(gen); refs.append([gold])

        pred_rows.append({
            "id": r["id"], "seeker": seeker, "pred": gen, "gold": gold,
            "topic": r.get("meta",{}).get("topic")
        })

    # Metrics
    P,R,F1 = bertscore_max_multi_ref(hyps, refs)
    tox_score = tox.score(hyps)
    ppl = ref_ppl.ppl(hyps)

    ensure_dir(outdir)
    pd.DataFrame(pred_rows).to_csv(os.path.join(outdir, "esconv_predictions.csv"), index=False)
    with open(os.path.join(outdir, "esconv_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"bertscore": {"P":P,"R":R,"F1":F1}, "toxicity": tox_score, "perplexity": ppl}, f, indent=2)

    return {"bertscore": {"P":P,"R":R,"F1":F1}, "toxicity": tox_score, "perplexity": ppl}


# ------------------------
# CounselChat runner
# ------------------------
def run_counselchat(records, model, outdir, tox: ToxicityScorer, ref_ppl: PerplexityScorer):
    groups = defaultdict(list)
    for r in records:
        qid = r["group_id"]
        groups[qid].append(r)

    hyps, refs_all, rows = [], [], []
    for qid, items in tqdm(groups.items(), desc="CounselChat groups"):
        q = items[0]["input"]["question"]
        prompt = COUNSELCHAT_PROMPT.format(question=q)
        gen = model.generate(prompt)

        refs = [strip_html(it["target"]["reply"]) for it in items]
        hyps.append(gen); refs_all.append(refs)
        rows.append({"group_id": qid, "question": q, "pred": gen, "n_refs": len(refs)})

    P,R,F1 = bertscore_max_multi_ref(hyps, refs_all)
    tox_score = tox.score(hyps)
    ppl = ref_ppl.ppl(hyps)

    ensure_dir(outdir)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "counselchat_predictions.csv"), index=False)
    with open(os.path.join(outdir, "counselchat_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"bertscore": {"P":P,"R":R,"F1":F1}, "toxicity": tox_score, "perplexity": ppl}, f, indent=2)

    return {"bertscore": {"P":P,"R":R,"F1":F1}, "toxicity": tox_score, "perplexity": ppl}


# ------------------------
# DSM runner (multi-task)
# ------------------------
def run_dsm(records, model, outdir):
    hyps, trues, confs, rows = [], [], [], []

    for r in tqdm(records, desc="DSM"):
        subtype = r.get("subtype", "mcq")  # default to mcq if missing

        if subtype == "mcq":
            q = r["input"]["question"]
            choices = r["input"]["choices"]
            prompt = DSM_MCQ_PROMPT.format(
                question=q,
                choices="\n".join([f"- {c}" for c in choices])
            )
            ans = model.classify_json(prompt)
            pred = normspace(ans.get("label", ""))
            conf = ans.get("confidence", None)
            true = normspace(r["target"]["label"])

            hyps.append(pred); trues.append(true); confs.append(conf)
            rows.append({
                "id": r["id"], "subtype": subtype,
                "question": q, "choices": choices,
                "pred": pred, "true": true, "confidence": conf, "raw": ans.get("raw")
            })

        elif subtype == "explanation":
            q = r["input"]["question"]
            answer = r["target"]["label"]
            prompt = DSM_EXPLANATION_PROMPT.format(question=q, answer=answer)
            gen = model.generate(prompt)
            gold = r["target"].get("rationale", "")

            rows.append({
                "id": r["id"], "subtype": subtype,
                "question": q, "answer": answer,
                "pred": gen, "gold": gold
            })

        elif subtype == "vignette":
            vignette = r["input"]["vignette"]
            prompt = DSM_VIGNETTE_PROMPT.format(vignette=vignette)
            ans = model.classify_json(prompt)
            pred = normspace(ans.get("label", ""))
            rationale = ans.get("rationale") or ans.get("raw", "")
            true = normspace(r["target"]["label"])
            gold_rationale = r["target"].get("rationale", "")

            hyps.append(pred); trues.append(true); confs.append(ans.get("confidence"))
            rows.append({
                "id": r["id"], "subtype": subtype,
                "vignette": vignette,
                "pred": pred, "true": true,
                "rationale": rationale, "gold_rationale": gold_rationale
            })

    # Metrics
    report = {}
    if hyps and trues:
        labels_sorted = sorted(set(trues) | set(hyps))
        cls = compute_cls_metrics(trues, hyps, labels=labels_sorted)
        cal = expected_calibration_error(trues, hyps, confs, bins=10)
        report.update({"classification": cls, "calibration": cal})

    ensure_dir(outdir)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "dsm_predictions.csv"), index=False)
    with open(os.path.join(outdir, "dsm_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name or path (e.g., meta-llama/Llama-3-8b-instruct)")
    ap.add_argument("--imhi", help="path to imhi_test.jsonl")
    ap.add_argument("--esconv", help="path to esconv_test.jsonl")
    ap.add_argument("--counselchat", help="path to counselchat_test.jsonl")
    ap.add_argument("--dsm", help="path to dsm_test.jsonl")
    ap.add_argument("--outdir", default="mhbench/outputs")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--ref-ppl-model", default="gpt2")
    args = ap.parse_args()

    model = HFCausalWrapper(
        args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    tox = ToxicityScorer("original")
    ref_ppl = PerplexityScorer(args.ref_ppl_model)

    report = {}

    if args.imhi:
        imhi_records = list(read_jsonl(args.imhi))
        report["IMHI"] = run_imhi(imhi_records, model, os.path.join(args.outdir, "IMHI"))

    if args.esconv:
        esconv_records = list(read_jsonl(args.esconv))
        report["ESConv"] = run_esconv(esconv_records, model, os.path.join(args.outdir, "ESConv"), tox, ref_ppl)

    if args.counselchat:
        cc_records = list(read_jsonl(args.counselchat))
        report["CounselChat"] = run_counselchat(cc_records, model, os.path.join(args.outdir, "CounselChat"), tox, ref_ppl)

    if args.dsm:
        dsm_records = list(read_jsonl(args.dsm))
        report["DSM"] = run_dsm(dsm_records, model, os.path.join(args.outdir, "DSM"))

    ensure_dir(args.outdir)
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
