def encode_text(text, tokenizer, max_len):
    return tokenizer.encode(text).ids[:max_len]
