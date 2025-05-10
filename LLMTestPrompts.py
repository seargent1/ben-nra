import os
import openai

#OPEN AI KEY input

system_prompts = [
    #"You are a concise doctor. Provide brief responses.",
    #"You are an empathetic doctor. Show care in your responses.",
    #"You are a skeptical doctor. Question the patient's assumptions.",
    #"You are a curious doctor. Inquire about patient's symptom history.",
    "Assess and diagnose the patient presenting with double vision, difficulty climbing stairs, and upper limb weakness",
    "Evaluate and diagnose the patient presenting with gait and limb ataxia, and a significant medical history.",
    "Assess and diagnose the patient presenting with knee pain following recent increased physical activity.",
    "Assess and diagnose the patient presenting with a chest skin lesion."]

# Sample user input
user_input = "I have a headache and fever. I feel sick in the stomach."

# Function to test each system prompt
def test_prompts():
    for prompt in system_prompts:
        print(f"\nSystem Prompt: {prompt}")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            reply = response['choices'][0]['message']['content']
            print(f"Response: {reply}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_prompts()
