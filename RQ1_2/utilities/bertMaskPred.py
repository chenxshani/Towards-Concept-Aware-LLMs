import pandas as pd
import torch
import transformers


def predict_masked_sent(
        text: str, model: transformers.BertForMaskedLM, tokenizer: transformers.BertTokenizer, top_k=50):

    # Tokenize input
    text = "[CLS] %s [SEP]" % text
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    top_k_tokens = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        top_k_tokens.append(predicted_token)
    return top_k_tokens


