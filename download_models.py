from transformers import (
                             BertTokenizer, BertConfig, BertModel,
                             RobertaTokenizer, RobertaConfig, RobertaModel,
                             T5Tokenizer, T5ForConditionalGeneration,
                             AutoTokenizer, AutoModelForCausalLM,
                         )

MODELS = {
             'bert': 'bert-base-cased',
             'roberta': 'roberta-base',
             't5-small': 't5-small',
             'llama-3.2-3B': 'meta-llama/Llama-3.2-3B',
         }

def main():
    BertTokenizer.from_pretrained(MODELS['bert'])
    BertConfig.from_pretrained(MODELS['bert'])
    BertModel.from_pretrained(MODELS['bert'])

    RobertaTokenizer.from_pretrained(MODELS['roberta'])
    RobertaConfig.from_pretrained(MODELS['roberta'])
    RobertaModel.from_pretrained(MODELS['roberta'])

    T5Tokenizer.from_pretrained(MODELS['t5-small'])
    T5ForConditionalGeneration.from_pretrained(MODELS['t5-small'])

    AutoTokenizer.from_pretrained(MODELS['llama-3.2-3B'])
    AutoModelForCausalLM.from_pretrained(MODELS['llama-3.2-3B'])

if __name__ == "__main__":
    main()
