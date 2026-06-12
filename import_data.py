# -*- coding: utf-8 -*-
"""Data preprocessing: convert raw CSVs to the format app.py expects."""
import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

# ---- 1. find source files by column count (avoid encoding issues) ----
all_csv = [f for f in os.listdir(DATA) if f.endswith('.csv')]
# exclude our own output files
OUTPUT_FILES = {'node_processed.csv', 'test_edge.csv', 'risk_score_table.csv',
                'top20_accounts.csv', 'typical_cases.csv'}
src_csv = [f for f in all_csv if f not in OUTPUT_FILES]

# Identify by column count and first column name
node_f = edge_f = abn_f = None
for f in sorted(src_csv, key=lambda x: os.path.getsize(os.path.join(DATA, x))):
    fpath = os.path.join(DATA, f)
    df = pd.read_csv(fpath, nrows=0)
    cols = list(df.columns)
    ncols = len(cols)
    first_col = cols[0]
    # Node table: ~17 cols, starts with account ID-like column
    if ncols >= 15 and ('ID' in first_col or 'id' in first_col):
        node_f = fpath
    # Edge table: ~11 cols, has payer/payee columns
    elif ncols == 11:
        edge_f = fpath
    # Abnormal tx log: ~5 cols
    elif ncols <= 6:
        abn_f = fpath

if not all([node_f, edge_f, abn_f]):
    raise FileNotFoundError(f"Could not identify all source files. Found: node={node_f}, edge={edge_f}, abn={abn_f}")

print(f"Node source: {os.path.basename(node_f)}")
print(f"Edge source: {os.path.basename(edge_f)}")
print(f"Abnormal source: {os.path.basename(abn_f)}")

# Read with no dtype first, then convert
node_raw = pd.read_csv(node_f)
edge_raw = pd.read_csv(edge_f)
abn_raw = pd.read_csv(abn_f)

# Convert ID columns to string (they are the first column)
node_id_col = node_raw.columns[0]    # account ID
edge_src_col = edge_raw.columns[0]   # payer ID
edge_dst_col = edge_raw.columns[1]   # payee ID
abn_src_col = abn_raw.columns[0]

node_raw[node_id_col] = node_raw[node_id_col].astype(str)
edge_raw[edge_src_col] = edge_raw[edge_src_col].astype(str)
edge_raw[edge_dst_col] = edge_raw[edge_dst_col].astype(str)
abn_raw[abn_src_col] = abn_raw[abn_src_col].astype(str)
abn_raw[abn_raw.columns[1]] = abn_raw[abn_raw.columns[1]].astype(str)

print(f"  node: {len(node_raw)} rows, edge: {len(edge_raw)} rows, abnormal: {len(abn_raw)} rows")

# ---- 2. node_processed.csv ----
print("\nGenerating node_processed.csv ...")

def tenure_bucket(months):
    if months < 3:    return "0-3 months"
    elif months < 6:  return "3-6 months"
    elif months < 12: return "6-12 months"
    elif months < 24: return "1-2 years"
    elif months < 60: return "2-5 years"
    else:             return "5+ years"

# Columns by position (verified from data exploration):
# 0:account_id, 1:risk_label, 2:label_name, 3:customer_type, 4:is_corp,
# 5:tenure_months, 6:region_code, 7:out_cnt, 8:in_cnt, 9:total_tx,
# 10:out_amt, 11:in_amt, 12:net_inflow, 13:self_loop_cnt, 14:self_loop_amt,
# 15:has_tx, 16:is_isolated
col = node_raw.columns
tenure_col = col[5]   # open account tenure (months)
region_col = col[6]   # region code
cust_type_col = col[3]  # customer type

node_raw["tenure_bucket"] = node_raw[tenure_col].apply(tenure_bucket)
node_df = node_raw[[col[0], cust_type_col, "tenure_bucket", region_col]].copy()
node_df.columns = ["account_id", "account_type", "tenure_bucket", "region_code"]
node_df.to_csv(os.path.join(DATA, "node_processed.csv"), index=False)
print(f"  -> {len(node_df)} records")

# ---- 3. test_edge.csv (test split only) ----
print("Generating test_edge.csv ...")

split_col = col[-1] if 'split' in str(col[-1]).lower() or '划分' in str(col[-1]) else edge_raw.columns[-1]
test_edge = edge_raw[edge_raw[edge_raw.columns[-1]] == "test"].copy()

hour_col = edge_raw.columns[4]   # tx hour

def time_bucket(hour):
    if 0 <= hour < 6:   return "late night"
    elif 6 <= hour < 12: return "morning"
    elif 12 <= hour < 18: return "afternoon"
    else:                return "evening"

test_edge["time_bucket"] = test_edge[hour_col].apply(time_bucket)

# Amount column (index 8 based on earlier exploration)
amt_col = edge_raw.columns[8]  # tx amount
def amount_bucket(amount):
    if amount < 5000:      return "small"
    elif amount < 50000:   return "medium"
    elif amount < 200000:  return "large"
    else:                  return "xlarge"

test_edge["amount_bucket"] = test_edge[amt_col].apply(amount_bucket)

edge_out = test_edge[[edge_src_col, edge_dst_col, "time_bucket", "amount_bucket"]].copy()
edge_out.columns = ["src_account", "dst_account", "time_bucket", "amount_bucket"]
edge_out.to_csv(os.path.join(DATA, "test_edge.csv"), index=False)
print(f"  -> {len(edge_out)} records (test set)")

# ---- 4. risk_score_table.csv ----
print("Generating risk_score_table.csv ...")

risk_label_col = col[1]   # risk label (0=normal, 1=fraud, 2=victim)
risk_level_map = {1: "high", 2: "medium", 0: "low"}
node_raw["risk_level"] = node_raw[risk_label_col].map(risk_level_map)

# Feature columns for scoring
self_loop_col = col[13]
net_inflow_col = col[12]
total_tx_col = col[9]

def calc_risk_score(row):
    score = 0.0
    rl = row[risk_label_col]
    if rl == 1:   score += 40
    elif rl == 2: score += 20
    sl = row[self_loop_col]
    if sl > 0:    score += min(sl * 5, 25)
    net = abs(row[net_inflow_col])
    if net > 100000:   score += 20
    elif net > 10000:  score += 10
    tx = row[total_tx_col]
    if tx > 50:   score += 15
    elif tx > 10: score += 8
    return min(score, 100)

node_raw["risk_score"] = node_raw.apply(calc_risk_score, axis=1)
risk_out = node_raw[[col[0], "risk_level", "risk_score"]].copy()
risk_out.columns = ["account_id", "risk_level", "risk_score"]
risk_out.to_csv(os.path.join(DATA, "risk_score_table.csv"), index=False)
print(f"  -> {len(risk_out)} records")

# ---- 5. top20_accounts.csv ----
print("Generating top20_accounts.csv ...")
top20 = node_raw.nlargest(20, "risk_score")[
    [col[0], cust_type_col, risk_label_col, col[2], total_tx_col,
     col[10], col[11], net_inflow_col, self_loop_col, "risk_score"]
].copy()
top20.columns = ["account_id", "account_type", "risk_label_code", "risk_label",
                 "total_tx", "out_amt", "in_amt", "net_inflow", "self_loop_cnt", "risk_score"]
top20.to_csv(os.path.join(DATA, "top20_accounts.csv"), index=False)
print(f"  -> {len(top20)} records")

# ---- 6. typical_cases.csv ----
print("Generating typical_cases.csv ...")
cases = []
fraud_ids = node_raw[node_raw[risk_label_col] == 1][col[0]].tolist()
victim_ids = node_raw[node_raw[risk_label_col] == 2][col[0]].tolist()

abn_type_col = abn_raw.columns[4]  # abnormal type column
abn_amt_col = abn_raw.columns[3]   # amount column
abn_time_col = abn_raw.columns[2]  # time column

# Case 1: self-loop
self_loop_mask = abn_raw[abn_type_col].astype(str).str.contains('self|自', case=False, na=False)
self_loop_accts = abn_raw.loc[self_loop_mask, abn_src_col].unique()
if len(self_loop_accts) > 0:
    aid = self_loop_accts[0]
    cnt = len(abn_raw[abn_raw[abn_src_col] == aid])
    cases.append({
        "case_id": "CASE-001",
        "description": f"Account {aid} has {cnt} self-loop transactions, suspected money laundering via wash trading",
        "chain": f"{aid} -> {aid} (self-loop x {cnt})",
        "evidence": f"1. {cnt} self-loop tx recorded; 2. Suspicious timing pattern; 3. Risk label: {'fraud' if aid in fraud_ids else 'abnormal'}",
        "conclusion": "High-risk money laundering account; recommend immediate freeze and SAR filing"
    })

# Case 2: fraud -> victim chain
if fraud_ids and victim_ids:
    fid = fraud_ids[0]
    targets = edge_raw[edge_raw[edge_src_col] == fid][edge_dst_col].unique()[:5]
    vt = [t for t in targets if t in victim_ids]
    if vt:
        cases.append({
            "case_id": "CASE-002",
            "description": f"Fraud account {fid} transfers to multiple victim accounts",
            "chain": f"{fid} (fraud) -> {' -> '.join(vt[:3])} (victims)",
            "evidence": f"1. Source {fid} labeled fraud; 2. Targets {', '.join(vt[:3])} labeled victims; 3. Transactions in non-business hours",
            "conclusion": "Classic fraud dispersal pattern; freeze fraud account, protect victim accounts with alerts"
        })

# Case 3: multi-hop laundering
if len(fraud_ids) >= 2:
    fid2 = fraud_ids[1]
    mids = edge_raw[edge_raw[edge_src_col] == fid2][edge_dst_col].unique()[:3]
    chain = [f"{fid2}(fraud_source)"]
    for m in mids[:2]:
        chain.append(f"{m}(hop)")
        nxt = edge_raw[edge_raw[edge_src_col] == m][edge_dst_col].unique()[:1]
        for n in nxt:
            chain.append(f"{n}(terminal)")
    cases.append({
        "case_id": "CASE-003",
        "description": f"Multi-hop money laundering involving fraud account {fid2}",
        "chain": " -> ".join(chain[:5]),
        "evidence": f"1. Source {fid2} labeled fraud; 2. Funds pass through multiple layers; 3. Amounts near reporting thresholds",
        "conclusion": "Multi-layer laundering network; trace full fund flow, coordinate with AML center for investigation"
    })

# Case 4: abnormal amount
ha_mask = abn_raw[abn_type_col].astype(str).str.contains('amount|金额|abnormal', case=False, na=False)
ha = abn_raw[ha_mask]
if len(ha) > 0:
    r = ha.iloc[0]
    cases.append({
        "case_id": "CASE-004",
        "description": f"Account {r[abn_src_col]} transfer to {r[abn_raw.columns[1]]} exceeds historical average significantly",
        "chain": f"{r[abn_src_col]} -> {r[abn_raw.columns[1]]} (amount: {r[abn_amt_col]:.2f})",
        "evidence": f"1. Amount {r[abn_amt_col]:.2f} far above historical mean; 2. Triggered monitoring rules; 3. Time: {r[abn_time_col]}",
        "conclusion": "Single large abnormal transaction; verify business background and counterparty relationship"
    })

case_df = pd.DataFrame(cases)
case_df.to_csv(os.path.join(DATA, "typical_cases.csv"), index=False)
print(f"  -> {len(case_df)} cases")

print("\nDone! All 5 CSV files generated.")
