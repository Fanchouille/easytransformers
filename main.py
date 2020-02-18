# coding: utf-8
from transformers import pipeline


def main(model_path, text):
    fill_mask = pipeline(
        "fill-mask",
        model=model_path,
        tokenizer=model_path
    )
    result = fill_mask(text.replace("<mask>", fill_mask.tokenizer.mask_token))
    print(result)


if __name__ == '__main__':
    text = "Edmond Dantès est le comte de Monte -<mask>."
    models = ["flaubert-small-cased", "flaubert-large-cased", "camembert-base",
              "./montecristo_model/checkpoint-500/", "./montecristo_model/checkpoint-1000/"]
    for model in models:
        print("Model : {}".format(model))
        main(model, text)
