import os, json, re, time, sys
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types

# ─────────────────────────────────────────────
# CONFIG  —  paste your key here OR set env var
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
MODEL_NAME     = "gemini-1.5-flash"
INPUT_CSV      = "claims.csv"
HISTORY_CSV    = "user_history.csv"
EVIDENCE_CSV   = "evidence_requirements.csv"
SAMPLE_CSV     = "sample_claims.csv"
OUTPUT_CSV     = "output.csv"
IMAGE_BASE_DIR = "."
DELAY_SECONDS  = 1.5
MAX_RETRIES    = 3

# ─────────────────────────────────────────────
# ALLOWED OUTPUT VALUES
# ─────────────────────────────────────────────
RISK_FLAGS = [
    "blurry_image", "wrong_angle", "damage_not_visible",
    "cropped_or_obstructed", "claim_mismatch",
    "possible_manipulation", "text_instruction_present",
    "user_history_risk", "manual_review_required",
]

ISSUE_TYPES = {
    "car":     ["dent", "scratch", "crack", "shatter", "broken_part",
                "paint_damage", "water_damage", "burn", "none", "unknown"],
    "laptop":  ["crack", "shatter", "dent", "scratch", "broken_part",
                "stain", "water_damage", "burn", "hinge_damage", "none", "unknown"],
    "package": ["crushed_packaging", "torn_packaging", "water_damage",
                "missing_item", "broken_seal", "stain", "burn", "none", "unknown"],
}

OBJECT_PARTS = {
    "car":     ["front_bumper", "rear_bumper", "windshield", "door", "hood",
                "trunk", "side_mirror", "headlight", "taillight", "roof",
                "wheel", "interior", "unknown"],
    "laptop":  ["screen", "keyboard", "trackpad", "body", "hinge",
                "port", "battery", "corner", "bottom_cover", "webcam", "unknown"],
    "package": ["outer_box", "box", "package_corner", "package_side",
                "seal", "contents", "label", "unknown"],
}


# ─────────────────────────────────────────────
# LOAD REFERENCE FILES
# ─────────────────────────────────────────────
def load_reference_data():
    history_lookup = {}
    if Path(HISTORY_CSV).exists():
        uh = pd.read_csv(HISTORY_CSV)
        for _, row in uh.iterrows():
            history_lookup[row["user_id"]] = {
                "past_claim_count":         int(row.get("past_claim_count", 0)),
                "accept_claim":             int(row.get("accept_claim", 0)),
                "rejected_claim":           int(row.get("rejected_claim", 0)),
                "manual_review_claim":      int(row.get("manual_review_claim", 0)),
                "last_90_days_claim_count": int(row.get("last_90_days_claim_count", 0)),
                "history_flags":            str(row.get("history_flags", "none")),
                "history_summary":          str(row.get("history_summary", "")),
            }
    else:
        print(f"[WARN] {HISTORY_CSV} not found — user history skipped.")

    evidence_lookup = {"all": [], "car": [], "laptop": [], "package": []}
    if Path(EVIDENCE_CSV).exists():
        er = pd.read_csv(EVIDENCE_CSV)
        for _, row in er.iterrows():
            obj   = str(row["claim_object"]).strip().lower()
            entry = {
                "id":         row["requirement_id"],
                "applies_to": row["applies_to"],
                "minimum":    row["minimum_image_evidence"],
            }
            if obj in evidence_lookup:
                evidence_lookup[obj].append(entry)
    else:
        print(f"[WARN] {EVIDENCE_CSV} not found — evidence requirements skipped.")

    return history_lookup, evidence_lookup


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────
def build_prompt(user_claim, claim_object, img_count, user_hist, ev_reqs):
    issues = ", ".join(ISSUE_TYPES.get(claim_object, ISSUE_TYPES["car"]))
    parts  = ", ".join(OBJECT_PARTS.get(claim_object, OBJECT_PARTS["car"]))
    flags  = ", ".join(RISK_FLAGS)

    hist_block = ""
    if user_hist:
        hist_block = f"""
USER CLAIM HISTORY:
  Past claims         : {user_hist['past_claim_count']}
  Accepted            : {user_hist['accept_claim']}
  Rejected            : {user_hist['rejected_claim']}
  Manual reviews      : {user_hist['manual_review_claim']}
  Claims last 90 days : {user_hist['last_90_days_claim_count']}
  History flags       : {user_hist['history_flags']}
  Summary             : {user_hist['history_summary']}
"""

    req_block = ""
    if ev_reqs:
        lines     = "\n".join(f"  [{r['id']}] '{r['applies_to']}': {r['minimum']}" for r in ev_reqs)
        req_block = f"\nEVIDENCE REQUIREMENTS:\n{lines}\n"

    return f"""You are a senior insurance claims analyst specialising in {claim_object} damage.

CUSTOMER CONVERSATION:
{user_claim}
{hist_block}{req_block}
EVIDENCE: {img_count} image(s) attached (labelled img_1, img_2, … in order).

TASK: Examine every image carefully alongside ALL context above, then return ONLY
a valid JSON object — no markdown fences, no extra text.

STRICT RULES:
1. Images are PRIMARY source of truth. Apply every evidence requirement above.
2. evidence_standard_met = true  → at least one image satisfies the relevant requirement.
   evidence_standard_met = false → images missing, blurry, wrong angle, or fail requirements.
3. valid_image = true  → at least one image is clear, relevant, correctly angled.
   valid_image = false → all images unusable.
4. claim_status:
   "supported"              → image confirms the described damage
   "contradicted"           → image shows something different
   "not_enough_information" → cannot verify due to image quality or angle
5. severity: "low" | "medium" | "high" | "critical" | "unknown" | "none"
   "none" only when contradicted with zero visible damage.
   "unknown" when image quality prevents judgement.
6. risk_flags: semicolon-separated from: {flags}
   Use "none" if no risks. Add "user_history_risk" if history flags contain it.
   ALWAYS add "manual_review_required" when ANY other flag is present.
7. supporting_image_ids: "none" OR "img_1" OR "img_1;img_2" etc.
8. issue_type : ONE from: {issues}
9. object_part: ONE from: {parts}

RETURN THIS JSON ONLY:
{{
  "evidence_standard_met": true or false,
  "evidence_standard_met_reason": "1-2 sentences referencing requirements and images",
  "risk_flags": "none or semicolon-separated flags",
  "issue_type": "one value",
  "object_part": "one value",
  "claim_status": "supported | contradicted | not_enough_information",
  "claim_status_justification": "1-2 sentences grounded in images and history",
  "supporting_image_ids": "none or img_1 or img_1;img_2",
  "valid_image": true or false,
  "severity": "low | medium | high | critical | unknown | none"
}}"""


# ─────────────────────────────────────────────
# IMAGE LOADER
# ─────────────────────────────────────────────
def load_images(image_paths_str: str):
    """Return list of (bytes, mime_type) tuples and total path count."""
    paths  = [p.strip() for p in str(image_paths_str).split(";") if p.strip()]
    images = []
    for rel in paths:
        full = Path(IMAGE_BASE_DIR) / rel
        if not full.exists():
            print(f"    [WARN] Missing image: {full}")
            continue
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png",  ".webp": "image/webp",
                ".gif": "image/gif"}.get(full.suffix.lower(), "image/jpeg")
        images.append((full.read_bytes(), mime))
    return images, len(paths)


# ─────────────────────────────────────────────
# GEMINI CALL  (new google-genai SDK)
# ─────────────────────────────────────────────
FALLBACK = {
    "evidence_standard_met":        False,
    "evidence_standard_met_reason": "Processing error — manual review needed.",
    "risk_flags":                   "manual_review_required",
    "issue_type":                   "unknown",
    "object_part":                  "unknown",
    "claim_status":                 "not_enough_information",
    "claim_status_justification":   "Could not process this claim automatically.",
    "supporting_image_ids":         "none",
    "valid_image":                  False,
    "severity":                     "unknown",
}

def call_gemini(client, row: pd.Series, history_lookup: dict, evidence_lookup: dict) -> dict:
    claim_object = row["claim_object"]
    user_id      = row["user_id"]
    user_hist    = history_lookup.get(user_id, {})
    ev_reqs      = evidence_lookup.get("all", []) + evidence_lookup.get(claim_object, [])

    images, img_count = load_images(row["image_paths"])
    prompt = build_prompt(row["user_claim"], claim_object, img_count, user_hist, ev_reqs)

    # Build content parts: text prompt + image bytes
    content_parts = [prompt]
    for img_bytes, mime in images:
        content_parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=content_parts,
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```\s*$", "", raw).strip()
            result = json.loads(raw)

            # Normalise booleans
            for field in ("evidence_standard_met", "valid_image"):
                if isinstance(result.get(field), str):
                    result[field] = result[field].strip().lower() == "true"

            # Merge history flags
            hist_flags = str(user_hist.get("history_flags", "none")).strip().lower()
            if hist_flags and hist_flags != "none":
                existing     = str(result.get("risk_flags", "none")).strip().lower()
                existing_set = set(existing.split(";")) if existing != "none" else set()
                history_set  = set(hist_flags.split(";"))
                merged       = existing_set | history_set
                merged.discard("none")
                if merged - {"manual_review_required"}:
                    merged.add("manual_review_required")
                result["risk_flags"] = ";".join(sorted(merged)) if merged else "none"

            return result

        except json.JSONDecodeError as e:
            print(f"    [JSON ERR attempt {attempt}] {e}")
        except Exception as e:
            print(f"    [API  ERR attempt {attempt}] {e}")

        time.sleep(2 ** attempt)

    print("    [FAIL] All retries exhausted — using fallback.")
    return FALLBACK.copy()


# ─────────────────────────────────────────────
# EVALUATE vs SAMPLE
# ─────────────────────────────────────────────
def evaluate(output_df: pd.DataFrame):
    if not Path(SAMPLE_CSV).exists():
        print(f"\n[SKIP] {SAMPLE_CSV} not found.")
        return
    sample     = pd.read_csv(SAMPLE_CSV)
    score_cols = ["claim_status", "severity", "issue_type",
                  "object_part", "valid_image", "evidence_standard_met"]
    merged     = sample[["user_id"] + score_cols].merge(
        output_df[["user_id"] + score_cols], on="user_id", suffixes=("_exp", "_pred"))
    total      = len(merged)
    print("\n── Evaluation vs sample_claims.csv ──")
    for col in score_cols:
        ok = (merged[f"{col}_exp"].astype(str).str.strip().str.lower() ==
              merged[f"{col}_pred"].astype(str).str.strip().str.lower()).sum()
        print(f"  {col:<28} {ok:>2}/{total}  ({100*ok/total:.0f}%)")
    perfect = sum(
        all(str(merged.at[i, f"{c}_exp"]).strip().lower() ==
            str(merged.at[i, f"{c}_pred"]).strip().lower() for c in score_cols)
        for i in merged.index)
    print(f"\n  Perfect rows : {perfect}/{total}")


# ─────────────────────────────────────────────
# SINGLE-ROW TEST
# ─────────────────────────────────────────────
def test_single_row(history_lookup, evidence_lookup):
    print("\n── Single-row test ──")
    client = genai.Client(api_key=GEMINI_API_KEY)
    df     = pd.read_csv(INPUT_CSV)
    row    = df.iloc[0]
    print(f"Testing: {row['user_id']} | {row['claim_object']}")
    result = call_gemini(client, row, history_lookup, evidence_lookup)
    print(json.dumps(result, indent=2))


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(history_lookup, evidence_lookup):
    print("=" * 54)
    print("  Multi-Modal Claims Processor  |  Gemini 1.5 Flash")
    print("=" * 54)

    client = genai.Client(api_key=GEMINI_API_KEY)
    df     = pd.read_csv(INPUT_CSV)

    print(f"Claims        : {len(df)} rows | {df['claim_object'].value_counts().to_dict()}")
    print(f"User histories: {len(history_lookup)} users loaded")
    print(f"Evidence reqs : {sum(len(v) for v in evidence_lookup.values())} rules loaded\n")

    OUT_COLS = [
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part",
        "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity",
    ]

    results = []
    for i, row in df.iterrows():
        uid  = row["user_id"]
        obj  = row["claim_object"]
        hist = history_lookup.get(uid, {})
        print(f"[{i+1:02d}/{len(df)}] {uid}  |  {obj}"
              + (f"  | hist={hist.get('history_flags')}" if hist else ""))
        res = call_gemini(client, row, history_lookup, evidence_lookup)
        print(f"         status={res.get('claim_status'):<26}"
              f"severity={res.get('severity'):<8}  issue={res.get('issue_type')}")
        results.append({c: res.get(c) for c in OUT_COLS})
        time.sleep(DELAY_SECONDS)

    out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅  {len(out_df)} rows saved → {OUTPUT_CSV}")
    print("\n── claim_status ──")
    print(out_df["claim_status"].value_counts().to_string())
    print("\n── severity ──")
    print(out_df["severity"].value_counts().to_string())
    evaluate(out_df)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    history_lookup, evidence_lookup = load_reference_data()
    if "--test" in sys.argv:
        test_single_row(history_lookup, evidence_lookup)
    else:
        run_pipeline(history_lookup, evidence_lookup)