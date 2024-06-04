import os
from transformers import AutoModelForCausalLM 
from transformers import AutoTokenizer
import torch

is_cuda_available = torch.cuda.is_available()
device = torch.device(
            "cuda" if is_cuda_available else "cpu")

model_path_hf = "mistralai/Mistral-7B-Instruct-v0.2"
model_path_hf = "meta-llama/Meta-Llama-3-8B-Instruct"

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
       <|begin_of_text|><|start_head_id|>system<|end_header_id|>
    You are a text classification bot.
    Your task is to assess intent and categorize the input
    text into one of the following predefined categories:

    2: Experiencer - Patient / default
    1: Experiencer - Family
    0: Not applicable

    Explanation of labels:
    Label 2 (patient / default) is the class where the context strongly indicates that the given medical entity is for the patient. The text will not explicitly contain mention that it is for the patient, you have to infer it.
    Label 1 (family) is the class where the context clearly indicates that the given medical entity is for the family.
    Label 0 (not applicable) is when the input data does is not applicable to the category.

    You will only respond with the predefined category.
    Do not provide explanations or notes.

    Format of input: 'tokens', 'medical entity'. Classify the text for the medical entity in the tokens.<|eot_id|>
    <|start_head_id|>user<|end_header_id|>
    Input: {text} <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

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

#with open("llm_few_exp_mistral_results.txt","w") as f:
#    f.write(f"ACCURACY: f{correct/(correct+wrong)}")
#    f.close()
