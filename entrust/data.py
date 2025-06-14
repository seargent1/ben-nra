import json 
import pandas 

#TODO ~ change to unique path 
data_path = "/Users/bencliu/2025_Summer/entrust/dataset/ENTRUST_All_Cases_Data-20250502.json"


""" 
dict_keys(['physicalExam', 'generalInfo', 'id', 'organization', 'copiedFrom', 'dispoSettings', 'vitalSigns', 'patientImage', 'createTime', 'gameSettings', 'orQuestions', 'caseOrders'])
- physicalExam: List of actions re. physical exams with points
- generalInfo: Capture into OSCE format (patient history)
- dispoSettings: List of actions re. disposition with points
- vitalSigns: Capture with generalInfo
- patientImage: Extract as metadata (includes demographics)

Broad classes:
1. Patient information (pt agent)
2. Actions (Dispo, physical exam)
3. Backend info:
- unique IDs
- questions at the very end

"""

"""
"""

def helper_physical_exam(input_dict):
    pass 

def playground():
    with open(data_path, "r") as f:
        data = json.load(f)
        pe_data = helper_physical_exam(data[0]['physicalExam'])
        breakpoint() 

if __name__ == "__main__":
    playground()