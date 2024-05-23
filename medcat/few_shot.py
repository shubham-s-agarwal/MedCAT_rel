import os
from transformers import AutoModelForCausalLM 
from transformers import AutoTokenizer
import torch


is_cuda_available = torch.cuda.is_available()
device = torch.device(
            "cuda" if is_cuda_available else "cpu")

model_path_hf = "mistralai/Mistral-7B-Instruct-v0.2"
model_path_hf = "meta-llama/Meta-Llama-3-8B"

model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path_hf,
            device_map='cuda',
            token = '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',
            torch_dtype=torch.float16
            )



model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path_hf,token='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')

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.65,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=100
)
mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate


examples = [{"text":"'HISTORY: , A 59-year-old male presents in followup after being evaluated and treated as an in-patient by Dr. X for acute supraglottitis with airway obstruction and parapharyngeal cellulitis and peritonsillar cellulitis, admitted on 05/23/2008, discharged on 05/24/2008.', 'airway obstruction'","label":"0"},{"text":"'She has had some decrease in her appetite, although her weight has been stable.  She has had no fever, chills, or sweats.  Activity remains good and she has continued difficulty with depression associated with type 1 bipolar disease.  She had a recent CT scan of the chest and abdomen.  The report showed the following findings.  In the chest, there was a small hiatal hernia and a calcification in the region of the mitral valve.', 'depression'","label":"0"},{"text":"'PAST MEDICAL HISTORY:,1.  Hypertension.,2.  Depression.,3.  Myofascitis of the feet.,4.  Severe osteoarthritis of the knee.,5.  Removal of the melanoma from the right thigh in 1984.,6.  Breast biopsy in January of 1997, which was benign.,7.  History of Holter monitor showing ectopic beat.  Echocardiogram was normal.' , 'osteoarthritis of the knee'","label":"0"},{"text":"'Skin and Lymphatics:  Examination of the skin does not reveal any additional scars, rashes, spots or ulcers.  No significant lymphadenopathy noted.,Spine:  Examination shows lumbar scoliosis with surgical scar with no major tenderness.' , 'lymphadenopathy'","label":"1"},{"text":"'She has no chest pain, palpitations, coughing or wheezing.  She does get shortness of breath, no hematuria, dysuria, arthralgias, myalgias, rashes, jaundice, bleeding or clotting disorders.  The rest of the system review is negative as per the HPI.','clotting disorders'","label":"1"}]

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
    ### TASK: Text Classification Task
    ### INSTRUCTION: Act as a text classification expert. You will be provided with examples of data. Using these examples as a reference, classify the provided data input into the following labels: Label 1: Status - Confirmed, Label 2: Status - Other. Label 1 (confirmed) is the class where the context strongly suggests that the patient has or had the given medical entity. Label 2 (other) is the class where the context clearly indicates the absence of confirmation for the given medical entity.
    INPUT: You are required to classify the following examples. They are provided in the format: 'tokens', 'medical entity'.
    ### Answer:
"""
)

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=final_prompt)

data_ = eval(open("metacat_.txt").read())

from tqdm import tqdm
class_mapping = {"Other":"2", "Confirmed":"1"}
correct = 0
wrong = 0
pbar = tqdm(total=len(data_))
for sample in data_:
    txt = sample[0]
    to_send = f'''"{txt}" , "{sample[1]}"  '''

    # {sample[0][sample[1] + 2]}
    input_text = {
        "input": to_send
    }
    #input_text= input_text.to(device)
    response = llm_chain(input_text)
    # print("SENT TEXT:",input_text)
    print("Comparing:",class_mapping[sample[-1]],response['text'].split(";")[0])
    if class_mapping[sample[-1]] in response['text'].split(";")[0]:
        correct += 1
    else:
        wrong += 1
    pbar.update(1)
    print("\nAccuracy", correct / (correct + wrong))

pbar.close()

print("Correct: ",correct," Wrong",wrong)
print("Accuracy",correct/(correct+wrong))



