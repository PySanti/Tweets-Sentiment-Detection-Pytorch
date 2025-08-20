from tokenizers import Tokenizer, trainers, models

def initialize_tokenizer(X):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    trainer = trainers.BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]  # Añade tokens especiales
    )
    tokenizer.train_from_iterator(X,trainer=trainer)
    return tokenizer


