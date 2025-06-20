import json
import pandas as pd
from pathlib import Path

# Path to the ENTRUST JSON data file (change according to your system)
ENTRUST_JSON = Path("/Users/raghav/Documents/GitHub/ben-nra/entrust/ENTRUST_All_Cases.json")

def extract_vitals(case):
    """
    Extract vital signs data from a case.
    
    Args:
        case (dict): A single case from the ENTRUST dataset
        
    Returns:
        dict: Dictionary containing vital signs values and their changes
    """
    # Get initial vital signs
    start = case.get("vitalSigns", {}).get("start", {})
    # Get vital signs changes/updates
    update = case.get("vitalSigns", {}).get("update", {})

    return {
        # Initial vital signs values
        "temperature": float(start.get("temp")) if "temp" in start else None,
        "sbp": float(start.get("sbp")) if "sbp" in start else None,  # Systolic blood pressure
        "dbp": float(start.get("dbp")) if "dbp" in start else None,  # Diastolic blood pressure
        "heart_rate": float(start.get("hr")) if "hr" in start else None,
        "resp_rate": float(start.get("rr")) if "rr" in start else None,  # Respiratory rate
        
        # Changes in vital signs (deltas)
        "temperature_delta": float(update.get("temp", {}).get("change", 0)),
        "sbp_delta": float(update.get("sbp", {}).get("change", 0)),
        "dbp_delta": float(update.get("dbp", {}).get("change", 0)),
        "heart_rate_delta": float(update.get("hr", {}).get("change", 0)),
        "resp_rate_delta": float(update.get("rr", {}).get("change", 0)),
    }

def fallback_result_text(source):
    """
    Extract meaningful text from test results when direct text is not available.
    
    Args:
        source (dict): Result data that may contain text, ranges, or subfields
        
    Returns:
        str: Formatted text describing the result
    """
    if isinstance(source, dict):
        # Try to get direct text first
        text = source.get("text", "")
        if text:
            return text.strip()
        
        # If range exists, format that (e.g., "Expected range: 70-100 mg/dL")
        if "range" in source and isinstance(source["range"], dict):
            r = source["range"]
            if "min" in r and "max" in r:
                return f"Expected range: {r['min']}–{r['max']} {source.get('unit', '')}".strip()
        
        # Handle subfields (e.g., BMP panel with multiple components)
        if "subFields" in source:
            return ", ".join(
                f"{s['name']}: {s['range']['min']}–{s['range']['max']} {s.get('unit', '')}"
                for s in source["subFields"]
                if "range" in s and "min" in s["range"] and "max" in s["range"]
            )
    return ""

def add_action(rowlist, *, case_id, description, kind, points, vitals_dict, fallback=""):
    """
    Add an action row to the list of actions.
    
    Args:
        rowlist (list): List to append the action row to
        case_id (str): Unique identifier for the case
        description (str): Description of the action (may contain "action: result" format)
        kind (str): Type of action (physical_exam, disposition, order, diagnosis_selection)
        points (int): Points awarded for this action
        vitals_dict (dict): Vital signs data to include with this action
        fallback (str): Fallback text if no result is found in description
    """
    # Split description into action and result if it contains ":"
    if ":" in description:
        action, result = description.split(":", 1)
        action = action.strip()
        result = result.strip()
    else:
        action = description.strip()
        result = ""

    # Use fallback text only if result is still empty
    if not result and fallback:
        result = fallback.strip()

    # Create the row with all necessary fields
    row = {
        "case_id": case_id,
        "action_type": kind,
        "action": action,
        "result": result,
        "point_change": points,
    }
    # Add vital signs data to the row
    row.update(vitals_dict)
    rowlist.append(row)

def extract_actions_from_case(case):
    """
    Extract all actions from a single case and convert them to DataFrame rows.
    
    Args:
        case (dict): A single case from the ENTRUST dataset
        
    Returns:
        list: List of dictionaries, each representing one action row
    """
    rows = []
    cid = case.get("id", "unknown_case")
    vitals = extract_vitals(case)

    # 1. Extract physical examination actions
    for key, block in case.get("physicalExam", {}).items():
        # Get text from either initial or new examination
        desc_text = block.get("initial", {}).get("text") or block.get("new", {}).get("text") or ""
        # Get points from either score or cosecsaScore field
        points = block.get("score", 0) or block.get("cosecsaScore", 0)
        description = f"{key}: {desc_text}" if desc_text else key
        add_action(rows, case_id=cid, description=description, kind="physical_exam", points=points, vitals_dict=vitals)

    # 2. Extract disposition choices
    for loc, cfg in case.get("dispoSettings", {}).items():
        points = cfg.get("score", 0)
        description = f"Disposition choice: {loc}"
        add_action(rows, case_id=cid, description=description, kind="disposition", points=points, vitals_dict=vitals)

    # 3. Extract case orders (tests, procedures, etc.)
    for entry in case.get("caseOrders", []):
        info = entry.get("order", {})
        name = info.get("fullName") or info.get("name", "Unnamed order")
        points = info.get("defaultScore", 0)
        # Get fallback text for test results
        fallback = fallback_result_text(info.get("defaultResult", {}))
        add_action(rows, case_id=cid, description=name, kind="order", points=points, vitals_dict=vitals, fallback=fallback)

    # 4. Extract OR question selections (diagnosis choices)
    for q in case.get("orQuestions", []):
        for sel in q.get("selections", []):
            name = sel.get("content", "choice")
            explanation = sel.get("explanation", "")
            # Get points from either cosecsaScore or score field
            points = sel.get("cosecsaScore", sel.get("score", 0))
            description = f"OR selection: {name}"
            fallback = f"Explanation: {explanation}" if explanation else ""
            add_action(rows, case_id=cid, description=description, kind="diagnosis_selection", points=points, vitals_dict=vitals, fallback=fallback)

    return rows

# Main execution block
if __name__ == "__main__":
    # Load the ENTRUST cases from JSON file
    with ENTRUST_JSON.open() as f:
        entrust_cases = json.load(f)

    # Process all cases and collect action rows
    all_rows = []
    for case in entrust_cases:
        all_rows.extend(extract_actions_from_case(case))
        breakpoint()  # Debug breakpoint - remove this in production

    # Convert to DataFrame and save to CSV
    actions_df = pd.DataFrame(all_rows)
    actions_df.to_csv("entrust_actions_with_vitals.csv", index=False)
