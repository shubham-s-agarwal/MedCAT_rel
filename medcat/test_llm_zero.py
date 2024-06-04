import os
from transformers import AutoModelForCausalLM 
from transformers import AutoTokenizer
import torch

# os.environ['TRANSFORMERS_CACHE'] = '/scratch/users/k2370999/huggingface_models/cache/'
torch.cuda.empty_cache()
is_cuda_available = torch.cuda.is_available()
device = torch.device(
            "cuda" if is_cuda_available else "cpu")
#evice = torch.device("cpu")

model_path_hf = model_path_hf = "mistralai/Mistral-7B-Instruct-v0.2"
#model_path_hf = "meta-llama/Meta-Llama-3-8B-Instruct"


model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path_hf,
       #     device_map='cuda',
            token = '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',
            torch_dtype=torch.float16)

#print("model loaded, moving to GPU")
#model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path_hf)

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
print("trying to load pipeline")
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=10,
    device=0)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
print("Pipeline loaded successfully")
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt_template = """
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
    Inquiry: {input}
    >>>
    Category: [/INST]
    """

#Label 2 (Patient) is the class where the context suggests that the input and/or medical entity is experienced by the patient. Label 1 (Family) is the class where the context suggests that the input and/or medical entity is experienced by the family of the patient. Label 0 (Other) is the not applicable class, used when the input is not applicable to label 1 and 2.


# Create prompt from prompt template
prompt = PromptTemplate(
    template=prompt_template, input_variables = ['input']
)

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

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
    count_mapping[sample[-1]]+=1
    # {sample[0][sample[1] + 2]}
    input_text = {
        "input": to_send
    }
    #input_text= input_text.to(device)
    response = llm_chain(input_text)
    #print("response", response)
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
#with open("llm_zero_exp_mistral_results.txt","w") as f:
#   f.write("ACCURACY:",correct/(correct+wrong))
#   f.close()

