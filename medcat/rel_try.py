from medcat.cdb import CDB
from medcat.config_rel_cat import ConfigRelCAT
from medcat.rel_cat import RelCAT
from medcat.utils.relation_extraction.tokenizer import TokenizerWrapperBERT,TokenizerWrapperLlama
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from medcat.utils.relation_extraction.models import BertModel_RelationExtraction,LlamaModel_RelationExtraction
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from transformers import AutoConfig

config = ConfigRelCAT()
config.general.cntx_left = 20
config.general.cntx_right = 20
config.general.model_name = "bert-base-uncased"
config.train.nclasses = 9
config.model.model_size= 2304
config.model.hidden_size= 256
config.train.batch_size = 64
config.train.nepochs = 4

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     config.general.device = device
# tokenizer = TokenizerWrapperBERT.load("./models/BERT_tokenizer_relation_extraction")

tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config.general.model_name, config=config), add_special_tokens=True)
# tokenizer = TokenizerWrapperLlama(AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config.general.model_name, token='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', config=config), add_special_tokens=True)

tokenizer.hf_tokenizers.add_tokens(["[s1]", "[e1]", "[s2]", "[e2]"], special_tokens=True)
config.general.annotation_schema_tag_ids= tokenizer.hf_tokenizers.convert_tokens_to_ids(["[s1]", "[e1]", "[s2]", "[e2]"])
tokenizer.hf_tokenizers.add_special_tokens({'pad_token': '[PAD]'})

print(tokenizer.hf_tokenizers.vocab_size, len(tokenizer.hf_tokenizers))
print(config.general["annotation_schema_tag_ids"])
# tokenizer.hf_tokenizers.save_pretrained("./models/BERT_tokenizer_relation_extraction")

cdb = CDB()
relCat = RelCAT(cdb, tokenizer, config=config)
model_config = BertConfig(num_hidden_layers=3)
# model_config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B",token='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',num_hidden_layers=3,pad_token_id=128260,vocab_size=128265)
print(model_config)

relCat.model = BertModel_RelationExtraction(pretrained_model_name_or_path=config.general.model_name,
                                                                        relcat_config=config,
                                                                        model_config=model_config)
relCat.model.bert_model.resize_token_embeddings(len(tokenizer.hf_tokenizers))

# relCat.model = LlamaModel_RelationExtraction(pretrained_model_name_or_path=config.general.model_name,
#                                                                         relcat_config=config,
#                                                                         model_config=model_config)
# relCat.model.llama_model.resize_token_embeddings(len(tokenizer.hf_tokenizers))
#

#
# peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=16,
#                          target_modules=["query", "value"], lora_dropout=0.2)
#
# relCat.model = get_peft_model(relCat.model, peft_config)
# relCat.model.print_trainable_parameters()
# print("PEFT configured!")

relCat.cdb = cdb
relCat.config = config
relCat.tokenizer = tokenizer

relCat.train(train_csv_path="/Users/k2370999/Downloads/RELCAT_TRAIN_CSV_v2.csv", test_csv_path="/Users/k2370999/Downloads/RELCAT_TEST_CSV_v2.csv", checkpoint_path="./test")

# relCat.train(train_csv_path="/Users/k2370999/Downloads/n2c2_train.tsv_export_downsampled", test_csv_path="/Users/k2370999/Downloads/n2c2_test.tsv_export_downsampled", checkpoint_path="./test")
