from pathlib import Path


class TokenizerTrainer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def train_tokenizer(self, data_folder, save_path, vocab_size=20000, min_frequency=2):
        paths = [str(x) for x in Path(data_folder).glob("**/*.txt")]

        # Customize training
        self.tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=min_frequency,
                             special_tokens=[
                                 "<s>",
                                 "<pad>",
                                 "</s>",
                                 "<unk>",
                                 "<mask>",
                             ])

        # Save files to disk
        self.tokenizer.save(save_path)
