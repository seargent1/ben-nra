import os
import sys

sys.path.append('/Users/raghav/Documents/agentclinic')

from agentclinic import DoctorAgent
from simulation import Scenario


#os library for OPEN AI Key input

# Define system prompts for 2 doctors
PROMPT_A = "You are a conservative doctor. Avoid experimental treatments."
PROMPT_B = "You are a bold doctor. You like innovative therapies even if they are not fully tested."

class FakeScenario(Scenario):
    def __init__(self):
        self.patient_id = "001"
        self.complaint = "Persistent fever and headache"
        self.condition = "Headache"
        self.metadata = {}
        self.history = ""
        self.state = {}
        self.agent_id = "DrAgent"

class CustomDoctorAgent(DoctorAgent):
    def __init__(self, scenario, backend_str="gpt-4", custom_prompt=None):
        super().__init__(scenario, backend_str)
        self.custom_prompt = custom_prompt

    def system_prompt(self):
        return self.custom_prompt if self.custom_prompt else super().system_prompt()

scenario = FakeScenario()
doc_a = CustomDoctorAgent(scenario, custom_prompt=PROMPT_A)
doc_b = CustomDoctorAgent(scenario, custom_prompt=PROMPT_B)

def simulate_doctor_chat():
    turn = "Patient reports headache and mild fever. Thoughts?"
    print(f"Dr. A: {turn}")
    reply_a = doc_a.inference(turn)
    print(f"\nDr. A: {reply_a}")

    reply_b = doc_b.inference(reply_a)
    print(f"\nDr. B: {reply_b}")

if __name__ == "__main__":
    simulate_doctor_chat()
