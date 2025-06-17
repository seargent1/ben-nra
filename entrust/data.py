import json
import pandas as pd
from pandas import json_normalize

#TODO ~ change to unique path 
data_path = "/Users/raghav/Documents/GitHub/ben-nra/entrust/ENTRUST_All_Cases.json"

""" 
dict_keys(['physicalExam', 'generalInfo', 'id', 'organization', 'copiedFrom', 'dispoSettings', 'vitalSigns', 'patientImage', 'createTime', 'gameSettings', 'orQuestions', 'caseOrders'])
- physicalExam: List of actions re. physical exams with points
- generalInfo: Capture into OSCE format (patient history)
- dispoSettings: List of actions re. disposition with points
- vitalSigns: Capture with generalInfo
- patientImage: Extract as metadata (includes demographics)
- orQuestions: questions regarding the patient for potential diagnoses
- caseOrders: list of treatments,tests, or procedures ordered for patient


Broad classes:
1. Patient information (pt agent)
2. Actions (Dispo, physical exam)
3. Backend info:
- unique IDs
- questions at the very end

"""

"""
osce = {
    "OSCE_Examination": {
        "Objective_for_Doctor": "",  # Task for doctor (e.g., assess weakness)
        "Patient_Actor": {
            "Demographics": "",  # Age, gender
            "History": "",  # Narrative of symptoms
            "Symptoms": {
                "Primary_Symptom": "",  # Main complaint
                "Secondary_Symptoms": []  # Other symptoms
            },
            "Past_Medical_History": "",  # Prior illnesses
            "Social_History": "",  # Lifestyle, habits
            "Review_of_Systems": ""  # Other symptoms checked or denied
        },
        "Physical_Examination_Findings": {
            "Vital_Signs": {
                "Temperature": "", "Blood_Pressure": "", "Heart_Rate": "", "Respiratory_Rate": ""
            },  # Vitals
            "Neurological_Examination": {
                "Cranial_Nerves": "", "Motor_Strength": "", "Reflexes": "", "Sensation": ""
            }  # Neuro exam
        },
        "Test_Results": {
            "Blood_Tests": {},  # Lab tests
            "Electromyography": {"Findings": ""},  # EMG results
            "Imaging": {"Chest_CT": {"Findings": ""}}  # Imaging findings
        },
        "Correct_Diagnosis": ""  # Final diagnosis
    }
}
"""

"""
update demographics for patient actor subset in the osce category from patient image key
- age
- gender
- patient name
"""
def enrich_demographics(patient_image):
    if not isinstance(patient_image, dict):
        return "Demographics not explicitly stated."
    
    age = patient_image.get("age")
    gender = patient_image.get("gender")

    parts = []
    if age:
        parts.append(f"{age}-year-old")
    if gender:
        parts.append(gender.lower())
    
    return f"{' '.join(parts)} patient" if parts else "Demographics not explicitly stated."



"""
create patient actor subset in the osce category from general info key 
- demographics
- history
- symptoms
- past medical history
- social history
- review of systems

"""
def patient_actor(general_info, patient_image):
    # Get history text, use chief complaint as fallback
    history_text = general_info.get("historyOfPresentIllness", "")
    if not history_text:
        history_text = general_info.get("chiefComplaint", "No history provided.")

    patient_name = general_info.get("patientName", "The patient")
    demographics = enrich_demographics(patient_image)


    symptoms_list = []
    if "chiefComplaint" in general_info:
        symptoms_list.append(general_info["chiefComplaint"])
    if "historyOfPresentIllness" in general_info:
        hpi = general_info["historyOfPresentIllness"]
        for p in hpi.split('.'):
            if any(x in p.lower() for x in ["pain", "weak", "loss", "injury", "vomit", "dizzy", "nausea", "blur", "confusion"]):
                symptoms_list.append(p.strip())

    primary = symptoms_list[0] if symptoms_list else "Not specified"
    secondary = symptoms_list[1:] if len(symptoms_list) > 1 else []

    past_med = general_info.get("pastMedicalHistory", "No significant past medical history.")
    social = general_info.get("socialHistory", "No relevant social history noted.")


    return {
        "Demographics": demographics,
        "History": history_text,
        "Symptoms": {
            "Primary_Symptom": primary,
            "Secondary_Symptoms": secondary
        },
        "Past_Medical_History": past_med,
        "Social_History": social,
        "Review_of_Systems": "Patient denies experiencing any chest pain, palpitations, shortness of breath, or recent infections."
    }

"""
create physical examination subset in the osce category from case orders key 
- vital signs
- neurological examination
- other findings

"""
def physical_examination(case_orders, physical_exam, vital_signs, or_questions):
    neuro_map = {
        "Sensory_Examination": "Sensory",
        "Sensation": "Sensory",
        "Sensory_Exam": "Sensory",
        "Motor_Examination": "Motor",
        "Motor_Function": "Motor",
        "Motor_Strength": "Motor",
        "Motor_Exam": "Motor",
        "Coordination_Tests": "Coordination",
        "Coordination_and_Gait": "Coordination",
        "Mental_Status_Examination": "Mental_Status",
        "Orientation": "Mental_Status",
        "Alert_and_oriented": "Mental_Status",
        "Upper_Extremities": "Motor",
        "Lower_Extremities": "Motor",
        "Limb_Examination": "Motor",
        "Strength_Assessment": "Motor",
        "Gait_Assessment": "Gait",
        "Cranial_Nerve_Examination": "Cranial_Nerves",
        "Pupils": "Cranial_Nerves",
        "Pupil_Reactivity": "Cranial_Nerves",
        "Eye_Movements": "Cranial_Nerves",
        "Facial_Expression": "Cranial_Nerves",
        "Speech": "Cranial_Nerves",
        "General": "Mental_Status",
        "Physical_Neurological_Examination": "Mental_Status",
        "Fontanelle": "Cranial_Nerves",
        "Mental_Status": "Mental_Status",
        "Consciousness_Level": "Mental_Status",
        "Gait": "Gait"
    }

    def extract_text(key):
        section = physical_exam.get(key, {})
        return (
            section.get("initial", {}).get("text") or
            section.get("new", {}).get("text") or
            ""
        ).strip()

    def classify_neuro_text(key, text):
        norm_key = neuro_map.get(key, key)
        text_lower = text.lower()
        if any(word in text_lower for word in ["ptosis", "pupil", "gaze", "cranial", "eye movement"]):
            return "Cranial_Nerve_Findings"
        elif any(word in text_lower for word in ["weakness", "motor", "movement", "strength"]):
            return "Motor_Findings"
        elif any(word in text_lower for word in ["alert", "conscious", "obtunded", "responds", "oriented", "speech", "arousable"]):
            return "Consciousness_Level"
        elif "reflex" in text_lower:
            return "Reflexes"
        elif any(word in text_lower for word in ["sensation", "numbness", "tingling", "sensory"]):
            return "Sensory_Findings"
        else:
            return "Other_Neurological_Findings"
    
    text = extract_text("neuro")
    if text:
        neuro_sections = {classify_neuro_text("neuro", physical_exam['neuro']['initial']['text']): physical_exam['neuro']['initial']['text']}
    else:
        neuro_sections = {}

    # Build Vital Signs
    start = vital_signs.get("start", {})
    sbp, dbp = start.get("sbp"), start.get("dbp")
    vitals = {
        "Temperature": f"{start.get('temp')}°C" if "temp" in start else "",
        "Blood_Pressure": f"{sbp}/{dbp} mmHg" if sbp and dbp else "",
        "Heart_Rate": f"{start.get('hr')} bpm" if "hr" in start else "",
        "Respiratory_Rate": f"{start.get('rr')} breaths/min" if "rr" in start else ""
    }

    # Gather findings
    other_sections = {}

    """
    for entry in case_orders:
        if not isinstance(entry, dict):
            continue
        if not entry.get("target", "").startswith("physicalExam."):
            continue

        key = entry["target"].split(".")[1]
        category = entry.get("category")
        text = extract_text(key)
        if not text:
            continue

        if category == "Neuro":
            neuro_type = classify_neuro_text(key, text)
            neuro_sections[neuro_type] = text
        else:
            other_sections[category] = text
    """

    # Add from orQuestions
    """
    if isinstance(or_questions, list):
        for q in or_questions:
            if not isinstance(q, dict):
                continue
            full_name = q.get("fullName", "").lower()
            text = q.get("defaultResult", {}).get("text", "").strip()
            if not text:
                continue
            if "neuro" in full_name or "neuro" in text.lower():
                neuro_type = classify_neuro_text(full_name, text)
                neuro_sections[neuro_type] = text
    """

    return {
        "Physical_Examination_Findings": {
            "Vital_Signs": vitals,
            "Neurological_Examination": neuro_sections,
        }
    }


"""
create objective for doctor subset in the osce category from general info key 
- objective for doctor
"""
def helper_general_info(input_dict):
    if isinstance(input_dict, str):
        return f"Assess and manage the patient presenting with: {input_dict}"
    # Get the chief complaint if available
    chief_complaint = input_dict.get('chiefComplaint', '')
    if chief_complaint:
        return f"Assess and manage the patient presenting with: {chief_complaint}"
    # Fallback to history of present illness
    history = input_dict.get('historyOfPresentIllness', '')
    if history:
        return f"Assess and manage the patient presenting with: {history}"
    return "Assess and manage the patient presenting with: No specific symptoms provided"

"""
create test results subset in the osce category from case orders key 
- blood tests
- electromyography
- imaging
"""
def extract_test_results(case_orders):
    test_results = {
        "Blood_Tests": {},
        "Electromyography": {"Findings": ""},
        "Imaging": {"Chest_CT": {"Findings": ""}}
    }
    for order_entry in case_orders:
        if not isinstance(order_entry, dict):
            continue
        test = order_entry.get("order", {})
        name = test.get("name", "").lower()
        result = test.get("defaultResult", {})
        text = result.get("text", "").strip()
        if test.get("categoryId") == "EI5WUNxOoQL8cz8I2qgt" and text:
            test_results["Blood_Tests"][test.get("name", "Unknown")] = text
        if test.get("categoryId") == "M3g8mfGdmzhAkY0s9bYi" and "chest" in name:
            test_results["Imaging"]["Chest_CT"]["Findings"] = text
        if "emg" in name or "electromyography" in name:
            test_results["Electromyography"]["Findings"] = text
    return test_results

"""
create correct diagnosis subset in the osce category from or questions key 
- correct diagnosis
"""
def extract_diagnosis(or_questions):
    for question in or_questions:
        for option in question.get("selections", []):
            if option.get("isAnswer", False):
                return option.get("content", "")
    return ""

"""
assemble osce case from keys
- general info
- case orders
- vital signs
- physical exam
- or questions
"""
def assemble_osce_case_from_keys(generalInfo, caseOrders, vitalSigns, physicalExam, orQuestions, patientImage):
    return {
        "OSCE_Examination": {
            "Objective_for_Doctor": helper_general_info(generalInfo),
            "Patient_Actor": patient_actor(generalInfo, patientImage),
            **physical_examination(caseOrders, physicalExam, vitalSigns, orQuestions),
            "Test_Results": extract_test_results(caseOrders),
            "Correct_Diagnosis": extract_diagnosis(orQuestions)
        }
    }

"""
playground
"""
def playground():
    with open(data_path, "r") as f:
        data = json.load(f)
        osce_rows = []
        for i in range(len(data)):
            case = data[i]  # Change to any index or iterate
            osce_row = assemble_osce_case_from_keys(
                case.get('generalInfo', {}),
                case.get('caseOrders', {}),
                case.get('vitalSigns', {}),
                case.get('physicalExam', {}),
                case.get('orQuestions', {}),
                case.get('patientImage', {})
            )
            osce_rows.append(osce_row)
            
    df = pd.DataFrame(osce_rows, columns=["OSCE_Examination"])
    df.to_csv("osce_formatted_cases.csv", index=False)

    with open("osce_formatted_cases.jsonl", "w") as f_out:
        for row in osce_rows:
            json.dump(row, f_out)
            f_out.write("\n")

    with open("osce_formatted_cases_ind_dictionaryformat.json", "w") as f_out:
        json.dump(osce_rows, f_out, indent=2)

if __name__ == "__main__":
    playground()