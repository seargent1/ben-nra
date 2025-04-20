# AgentClinic Documentation

This document provides a brief overview of the AgentClinic codebase (`agentclinic.py`), focusing on its structure and functionality.

## 1. Overview

AgentClinic simulates clinical interactions between AI agents (Doctor and Patient) to evaluate their diagnostic capabilities and potential biases. It supports various Large Language Models (LLMs) and datasets.

## 2. Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Set necessary API keys as environment variables or pass them via command-line arguments (e.g., `OPENAI_API_KEY`, `REPLICATE_API_TOKEN`, `ANTHROPIC_API_KEY`).

## 3. Core Components

### 3.1. `query_model` Function

- **Purpose**: Handles communication with various LLM backends (OpenAI, Replicate, Anthropic, local HuggingFace models).
- **Inputs**: Model identifier (`model_str`), prompt, system prompt, and optional parameters like timeout, image requests, etc.
- **Output**: Text response from the selected LLM.
- **Supported Models**: GPT-4 series, GPT-3.5, Claude 3.5 Sonnet, Llama 2/3, Mixtral, o1-preview, and HuggingFace models (via `HF_` prefix).

### 3.2. Scenario Classes (`Scenario`)

- **Examples**: `ScenarioMedQA`, `ScenarioMedQAExtended`, `ScenarioMIMICIVQA`, `ScenarioNEJM`, `ScenarioNEJMExtended`.
- **Purpose**: Represent individual clinical cases or scenarios. Each class parses specific data formats (e.g., from JSONL files) and provides structured access to patient information, examiner objectives, test results, physical exams, and the correct diagnosis.
- **Methods**: Typically include `patient_information()`, `examiner_information()`, `exam_information()`, `diagnosis_information()`.

### 3.3. Scenario Loader Classes (`ScenarioLoader`)

- **Examples**: `ScenarioLoaderMedQA`, `ScenarioLoaderMedQAExtended`, `ScenarioLoaderMIMICIV`, `ScenarioLoaderNEJM`, `ScenarioLoaderNEJMExtended`.
- **Purpose**: Load scenario data from corresponding JSONL files (`agentclinic_*.jsonl`).
- **Methods**: `__init__` (loads data), `sample_scenario` (returns a random scenario), `get_scenario` (returns a specific scenario by ID).

### 3.4. `PatientAgent` Class

- **Purpose**: Simulates the patient in the clinical interaction.
- **Initialization**: Takes a scenario object, LLM backend string (`backend_str`), and optional bias type (`bias_present`).
- **Key Methods**:
  - `inference_patient(question)`: Generates the patient's response to the doctor's question using the specified LLM, maintaining conversation history (`agent_hist`).
  - `generate_bias()`: Returns a string describing the active cognitive bias to be included in the system prompt, influencing the patient's behavior.
  - `system_prompt()`: Constructs the system prompt for the patient LLM, including role description, symptoms (from the scenario), and bias information.
  - `reset()`: Clears history and reloads patient symptoms for a new simulation.
  - `add_hist(hist_str)`: Adds external text (like measurement results) to the conversation history.

### 3.5. `DoctorAgent` Class

- **Purpose**: Simulates the doctor in the clinical interaction.
- **Initialization**: Takes a scenario object, LLM backend string (`backend_str`), maximum inference turns (`max_infs`), optional bias type (`bias_present`), and image request capability (`img_request`).
- **Key Methods**:
  - `inference_doctor(question, image_requested)`: Generates the doctor's response/question using the specified LLM, maintaining conversation history (`agent_hist`) and tracking inference count (`infs`). Supports multimodal input if `image_requested` is true.
  - `generate_bias()`: Returns a string describing the active cognitive bias for the doctor's system prompt.
  - `system_prompt()`: Constructs the system prompt for the doctor LLM, including role description, objectives (from the scenario), inference limits, instructions for requesting tests (`REQUEST TEST: [test]`) or images (`REQUEST IMAGES`), and bias information.
  - `reset()`: Clears history and reloads examiner information.

### 3.6. `MeasurementAgent` Class

- **Purpose**: Simulates a system providing results for medical tests requested by the doctor.
- **Initialization**: Takes a scenario object and LLM backend string (`backend_str`).
- **Key Methods**:
  - `inference_measurement(question)`: Uses an LLM to find and format the requested test results based on the information available in the scenario's `exam_information()`. Responds with "NORMAL READINGS" if the specific test isn't found.
  - `system_prompt()`: Defines the role and instructions for the measurement LLM.
  - `reset()`: Clears history and reloads available exam/test information.
  - `add_hist(hist_str)`: Adds dialogue turns to its internal history for context.

### 3.7. `compare_results` Function

- **Purpose**: Uses a separate 'moderator' LLM to compare the doctor's final diagnosis against the correct diagnosis from the scenario.
- **Inputs**: Doctor's diagnosis string, correct diagnosis string, moderator LLM identifier (`moderator_llm`).
- **Output**: "yes" or "no" indicating if the diagnoses match.

## 4. Datasets

The simulation uses datasets stored in JSONL format:

- `agentclinic_medqa.jsonl`: Based on MedQA.
- `agentclinic_medqa_extended.jsonl`: Extended MedQA cases.
- `agentclinic_mimiciv.jsonl`: Based on MIMIC-IV clinical cases.
- `agentclinic_nejm.jsonl`: Based on NEJM case challenges (includes images).
- `agentclinic_nejm_extended.jsonl`: Extended NEJM cases.

Select the dataset using the `--agent_dataset` argument (e.g., `MedQA`, `MedQA_Ext`, `MIMICIV`, `NEJM`, `NEJM_Ext`).

## 5. Biases

Cognitive biases can be introduced to influence agent behavior:

- **Doctor Biases**: `recency`, `frequency`, `false_consensus`, `confirmation`, `status_quo`, `gender`, `race`, `sexual_orientation`, `cultural`, `education`, `religion`, `socioeconomic`. Set via `--doctor_bias`.
- **Patient Biases**: `recency`, `frequency`, `false_consensus`, `self_diagnosis`, `gender`, `race`, `sexual_orientation`, `cultural`, `education`, `religion`, `socioeconomic`. Set via `--patient_bias`.

The `generate_bias()` method in each agent class provides the specific text added to the system prompt for the chosen bias.

## 6. Running Simulations

The `main` function orchestrates the simulation loop:

1.  Parses command-line arguments (`argparse`).
2.  Initializes API keys and loads the selected `ScenarioLoader`.
3.  Loops through the specified number of scenarios (`--num_scenarios`).
4.  For each scenario:
    - Initializes `PatientAgent`, `DoctorAgent`, and `MeasurementAgent` with specified LLMs and biases.
    - Runs the interaction loop for a set number of turns (`--total_inferences`).
    - Handles dialogue turns between Doctor and Patient.
    - If the Doctor requests a test (`REQUEST TEST`), invokes `MeasurementAgent`.
    - If the Doctor requests an image (`REQUEST IMAGES` and `--doctor_image_request` is True for NEJM datasets), passes image context to the Doctor LLM.
    - If the Doctor provides a diagnosis (`DIAGNOSIS READY`), calls `compare_results`.
    - Prints the dialogue and results.
5.  Reports overall accuracy.

**Example CLI Commands:**

- **Basic GPT-4o run:**
  ```bash
  python3 agentclinic.py --openai_api_key "YOUR_KEY" --doctor_llm gpt4o --patient_llm gpt4o
  ```
- **Run with biases (GPT-3.5 Doctor, GPT-4 Patient):**
  ```bash
  python3 agentclinic.py --openai_api_key "YOUR_KEY" --doctor_llm gpt3.5 --patient_llm gpt4 --patient_bias self_diagnosis --doctor_bias recency
  ```
- **Run with NEJM dataset and image requests (GPT-4o):**
  ```bash
  python3 agentclinic.py --openai_api_key "YOUR_KEY" --doctor_llm gpt4o --patient_llm gpt4o --agent_dataset NEJM --doctor_image_request True
  ```
- **Run with a local HuggingFace model (Mixtral):**
  ```bash
  python3 agentclinic.py --patient_llm "HF_mistralai/Mixtral-8x7B-v0.1" --moderator_llm "HF_mistralai/Mixtral-8x7B-v0.1" --doctor_llm "HF_mistralai/Mixtral-8x7B-v0.1" --measurement_llm "HF_mistralai/Mixtral-8x7B-v0.1"
  ```

Refer to `python3 agentclinic.py --help` for all available arguments.
