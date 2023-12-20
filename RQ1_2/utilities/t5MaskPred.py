from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import torch
import re


class T5:
    def __init__(self, t5_path="t5-large"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(t5_path)
        self.config = T5Config.from_pretrained(t5_path)
        self.mlm = T5ForConditionalGeneration.from_pretrained(t5_path, config=self.config).to(self.device)

    def _filter(self, output, end_token=["<extra_id_1>", ".", "!", "?"]):
        _txt = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # Filtering capital words as they indicate a new sentence
        first_cap = re.search("([A-Z])", _txt)
        if first_cap:
            _txt = _txt[:first_cap.regs[0][0]]
        # Filtering repeated words
        words_set = set()
        for w in _txt.split():
            if w in words_set:
                ind = [m.start() for m in re.finditer(w, _txt)]
                _txt = _txt[:ind[1]]
                break
            words_set.add(w)
        # Filtering end tokens
        for tok in end_token:
            if tok in _txt:
                _end_token_index = _txt.index(tok)
                return _txt[:_end_token_index]
        else:
            return _txt

    def get_top_k_predictions(self, masked_sen, topk, beams_multip=4, max_len=15):
        context_ids = self.tokenizer.encode_plus(masked_sen, #add_special_tokens=True,
                                                 return_tensors='pt')['input_ids'].to(self.device)

        outputs = self.mlm.generate(input_ids=context_ids,
                                    num_beams=topk*beams_multip,
                                    num_return_sequences=topk,
                                    max_length=max_len,
                                    min_length=5,
                                    eos_token_id=5,  # 14125, 22354
                                    pad_token_id=32098,
                                    forced_eos_token_id=32098,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    no_repeat_ngram_size=2,
                                    early_stopping=True
                                    )
        results = list(map(self._filter, outputs.sequences))
        return results, outputs.sequences_scores

