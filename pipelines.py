from transformers import pipeline


def main():
    generator = pipeline(
        "text-generation", model="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    print(
        generator(
            "In this course, we will teach you how to",
            max_length=30,
            num_return_sequences=2,
            pad_token_id=generator.tokenizer.eos_token_id,
            truncation=True,
        )
    )


if __name__ == "__main__":
    main()
