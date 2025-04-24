import openai
import json
import random
import os
import re
import datetime

API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = API_KEY

DATA_FILE = "agentclinic_nejm.jsonl"
LOG_FILE = "demo_log.json"
DOCTOR_MODEL = "gpt-4.1-nano"
MODERATOR_MODEL = "gpt-4.1-nano"
NUM_INTERACTION_TURNS = 3

# OpenAI API call
def query_llm(model_name, prompt, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=150,
        )
    answer = response.choices[0].message.content
    return re.sub(r"\s+", " ", answer).strip()

# load and pick a scenario
def load_and_select_scenario(filepath=DATA_FILE):
    with open(filepath, "r") as f:
        scenarios = [json.loads(line) for line in f]
    return random.choice(scenarios)

# build case summary
def get_case_summary(scenario_dict):
    question = scenario_dict.get("question", "N/A")
    patient_info = scenario_dict.get("patient_info", "N/A")
    physical_exams = scenario_dict.get("physical_exams", "N/A")
    return f"Question:\n{question}\n\nPatient Information:\n{patient_info}\n\nPhysical Examination Findings:\n{physical_exams}"

# retrieve correct diagnosis
def get_correct_diagnosis(scenario_dict):
    answers = scenario_dict.get("answers", [])
    for answer in answers:
        if answer.get("correct"):
            return answer.get("text", "Correct diagnosis not found")
    return "Correct diagnosis not specified in data"

# save run results
def save_run_data(run_data, filepath=LOG_FILE):
    log_data = []
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            with open(filepath, "r") as f:
                log_data = json.load(f)
            if not isinstance(log_data, list):
                log_data = []
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filepath}. Starting fresh log.")
            log_data = []

    log_data.append(run_data)

    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=4)

# demo entry point
def run_demo():
    scenario = load_and_select_scenario()
    case_summary = get_case_summary(scenario)
    correct_diagnosis = get_correct_diagnosis(scenario)
    question = scenario.get("question", "Question not found")

    print(f">>> Scenario: {question}")

    sys_prompts = {
        "Doctor 1": (
            "You are Doctor 1, collaborating with Doctor 2 on a diagnosis. "
            "Review the case summary and the ongoing discussion. Focus on diagnostic reasoning. Be concise. "
            "If it's your first turn, share initial thoughts. Otherwise, respond to Doctor 2."
        ),
        "Doctor 2": (
            "You are Doctor 2, collaborating with Doctor 1. "
            "Review the case summary and the ongoing discussion. Respond to Doctor 1, "
            "offer alternatives, or build on their points. Be concise."
        )
    }
    moderator_sys_prompt = (
        "You are an expert medical moderator. Compare the 'Proposed Diagnosis' with the 'Correct Diagnosis'. "
        "Determine if they refer to the same medical condition, even if phrased differently. "
        "Respond ONLY with 'Yes' or 'No'."
    )

    conversation_history = f"Case Summary:\n{case_summary}\n\n"
    doctors = ["Doctor 1", "Doctor 2"]
    last_response = ""
    last_speaker_index = -1

    total_turns = NUM_INTERACTION_TURNS * len(doctors)
    for turn in range(total_turns):
        current_doctor_index = turn % len(doctors)
        current_doctor = doctors[current_doctor_index]

        if turn == 0:
            prompt = f"{conversation_history}What are your initial thoughts?"
        else:
            other_doctor = doctors[(current_doctor_index - 1 + len(doctors)) % len(doctors)]
            prompt = f"{conversation_history}\nBased on the case and {other_doctor}'s last statement ({last_response}), what is your response?"

        response = query_llm(DOCTOR_MODEL, prompt, sys_prompts[current_doctor])
        conversation_history += f"{current_doctor}: {response}\n"
        last_response = response
        last_speaker_index = current_doctor_index

        print(f">>> {current_doctor} responded")

    last_speaking_doctor = doctors[last_speaker_index]
    final_prompt = (
        f"{conversation_history}\nBased on the entire discussion, "
        f"what is your single, most likely final diagnosis? State it clearly after 'Final Diagnosis:'."
    )
    final_diagnosis_attempt = query_llm(DOCTOR_MODEL, final_prompt, sys_prompts[last_speaking_doctor])

    proposed_diagnosis_text = final_diagnosis_attempt
    match = re.search(r"final diagnosis:(.*)", final_diagnosis_attempt, re.IGNORECASE)
    if match:
        proposed_diagnosis_text = match.group(1).strip()
        if not proposed_diagnosis_text:
             proposed_diagnosis_text = final_diagnosis_attempt

    print(f">>> Proposed Diagnosis: {proposed_diagnosis_text}")
    print(f">>> Correct Diagnosis: {correct_diagnosis}")

    moderator_prompt = (
        f"Correct Diagnosis: {correct_diagnosis}\n"
        f"Proposed Diagnosis: {proposed_diagnosis_text}\n\n"
        "Are these the same diagnosis?"
    )
    moderator_verdict = query_llm(MODERATOR_MODEL, moderator_prompt, moderator_sys_prompt)

    print(f">>> Moderator Verdict: {moderator_verdict}")
    result = "SUCCESS" if moderator_verdict.lower() == "yes" else "FAILURE"
    print(f">>> Result: {result}")

    run_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "doctor_model": DOCTOR_MODEL,
        "moderator_model": MODERATOR_MODEL,
        "num_turns_per_doctor": NUM_INTERACTION_TURNS,
        "question": question,
        "case_summary": case_summary,
        "correct_diagnosis": correct_diagnosis,
        "conversation_history": conversation_history,
        "proposed_diagnosis": proposed_diagnosis_text,
        "moderator_verdict": moderator_verdict,
        "result": result
    }

    save_run_data(run_data)

if __name__ == "__main__":
    # run script
    run_demo()