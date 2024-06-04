import os
from transformers import AutoModelForCausalLM 
from transformers import AutoTokenizer
import torch

is_cuda_available = torch.cuda.is_available()
device = torch.device(
            "cuda" if is_cuda_available else "cpu")

model_path_hf = "mistralai/Mistral-7B-Instruct-v0.2"
#model_path_hf = "meta-llama/Meta-Llama-3-8B"

model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path_hf,
            torch_dtype=torch.float16
            )

# model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path_hf)

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=10,
    device=0
)
mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate


examples = [{"text":"'care of Professor Riley at Queens Square as he had nerve conduction studies with them in order to rule out amyloid, he mentioned that his brother has amyloid realted neuropathy. He is also due to see Royal Free in about four weeks to discuss further options', 'amyloid'","label":"1"},{"text":"'Referred with possible history of recurrent pericarditis\n\n2. Family history of amyloidosis (uncle)\n\nMedication\n\nNil \n\nVisit Notes\n\nI had the pleasure of meeting Lee in the cardiac clinic today', 'amyloidosis'","label":"1"},{"text":"'High Ferritin 1512\nPlease see below for full set of bloods.\n \n \nJacob has been informed that legally he must not drive and must notify the DVLA of alcohol dependence. Licence will\nbe refused or revoked until a minimum of 1 year’s abstinence from alcohol consumption' , 'alcohol dependence'","label":"0"},{"text":"'diastolic dysfunction on echo).\n\n5. Seen by Prof Julian Gillmore at National Amyloidosis Centre, Royal Free Hospital: Fat biopsy consistent with TTR amyloidosis, genotype results awaited. SAP scan no visceral amyloid. DPD Jan 2021 strongly positive, Perugini 3' , 'TTR amyloidosis'","label":"0"},{"text":"'She has no chest pain, palpitations, coughing or wheezing. She does get shortness of breath, rashes, jaundice, bleeding or clotting disorders.','shortness of breath'","label":"2"},{"text":"'he felt slightly unwell just before seizure hence sat down\non the floor of the station, he is Unsure how long episode lasted. He had a previous seizure 20 days ago secondary to ETOH\nwithdrawal','seizure'","label":"2"}]

prompt_template = '''
Messages: {text}
Categories: {label}
'''

example_prompt = PromptTemplate(input_variables=["text","label"],template = prompt_template)


final_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix= """Text: {text} \nCategories: """,
    input_variables=["text"],
    prefix= """
    [INST]You are a text classification bot.
    Your task is to assess intent and categorize the input
    text after <<<>>> into one of the following predefined categories:

    2: Experiencer - Patient / default
    1: Experiencer - Family
    0: Not applicable

    Explanation of labels:
    Label 2 (patient / default) is the class where the context strongly indicates that the given medical entity is for the patient. The text will not explicitly contain mention that it is for the patient, you have to infer it.
    Label 1 (family) is the class where the context clearly indicates that the given medical entity is for the family.
    Label 0 (not applicable) is when the input data does is not applicable to the category.

    You will only respond with the predefined category.
    Do not provide explanations or notes.

    Format of input: 'tokens', 'medical entity'. Classify the text for the medical entity in the tokens.
    <<<
    Inquiry: {text}
    >>>
    Category: [/INST]
"""
)

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=final_prompt)

data_ = eval(open("formatted_data_exp_2.txt").read())

from tqdm import tqdm
class_mapping = {"Family":"1", "Patient":"2", "Other":"0"}
correct = 0
wrong = 0

count_mapping = {'0':0,'1':0,'2':0}
perf_mapping = {'0':0,'1':0,'2':0}


pbar = tqdm(total=len(data_))
for sample in data_:
    txt = sample[0]
    to_send = f'''"{txt}" , "{sample[1]}"  '''
    sample[-1] = str(sample[-1])
    # {sample[0][sample[1] + 2]}
    input_text = {
        "text": to_send
    }
    count_mapping[sample[-1]]+=1
    #print("SENDING",input_text)
    #input_text= input_text.to(device)
    response = llm_chain(input_text)
    #print("Response",response)
    print("Comparing:",sample[-1],response['text'].split(";")[0])

    try:
        if sample[-1] in response['text'].split(";")[0]:
            correct += 1
            perf_mapping[sample[-1]] +=1
        else:
            wrong += 1
  #      print("Comparing:",class_maping[sample[-1]],response['text'].split(";")[0])
    except:
        if sample[-1] in response['text'].split(";")[0]:
            correct +=1
            perf_mapping[sample[-1]] +=1
        else:
            wrong+=1
 #       print("Comparing:",sample[-1],response['text'].split(";")[0])
    pbar.update(1)
    print("\nAccuracy", correct / (correct + wrong))


pbar.close()

print("Correct: ",correct," Wrong",wrong)
print("Accuracy",correct/(correct+wrong))
print("Performance across class",perf_mapping)
print("Count across class",count_mapping)

with open("llm_few_exp_mistral_results.txt","w") as f:
    f.write(f"ACCURACY: f{correct/(correct+wrong)}")
    f.close()
