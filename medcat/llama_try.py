# # from transformers import LlamaForSequenceClassification, AutoConfig
# # config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B",token='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',num_hidden_layers=3)
# # model = LlamaForSequenceClassification.from_pretrained("meta-llama/Meta-Llama-3-8B",token='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',config=config)
# #
# # print(model)
#
# from transformers.models.auto.tokenization_auto import AutoTokenizer
# from medcat.utils.relation_extraction.tokenizer import TokenizerWrapperBERT,TokenizerWrapperLlama
# tokenizer = TokenizerWrapperLlama(AutoTokenizer.from_pretrained(pretrained_model_name_or_path="meta-llama/Meta-Llama-3-8B", token='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'), add_special_tokens=True)
# print(tokenizer.hf_tokenizers.pad_token)
# tokenizer.hf_tokenizers.add_tokens(["[s1]", "[e1]", "[s2]", "[e2]"], special_tokens=True)
# # config.general.annotation_schema_tag_ids= tokenizer.hf_tokenizers.convert_tokens_to_ids(["[s1]", "[e1]", "[s2]", "[e2]"])
# tokenizer.hf_tokenizers.add_special_tokens({'pad_token': '[PAD]'})
#
# print(tokenizer.hf_tokenizers.pad_token)
# print(tokenizer.hf_tokenizers.pad_token_id)
# print("NUM",len(tokenizer.hf_tokenizers))

batch_size = 3
import torch
tensor = torch.randn(2, 3, 4096)
print(tensor.shape)
tensor2 = tensor.view(batch_size, -1)
print(tensor2.shape)