import torch


def answer_question(question: str, context: str, model, tokenizer) -> str:
    tokenized_context = tokenizer.encode_plus(question, context)
    # tokenized_question = tokenizer.encode(question)
    # input_ids = tokenized_context.ids  +tokenized_question.ids[1:]
    input_ids = tokenized_context['input_ids']
    # token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
    # attention_mask = [1] * len(input_ids)
    attention_mask = tokenized_context['attention_mask']
    # padding_length = 384 - len(input_ids)
    # if padding_length > 0:
    #     input_ids = input_ids + ([0] * padding_length)
    #     attention_mask = attention_mask + ([0] * padding_length)
    # token_type_ids = token_type_ids + ([0] * padding_length)
    # token_type_ids_all.append(token_type_ids)
    output = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
    print(question)
    answer_words = tokenizer.convert_ids_to_tokens(
        input_ids[torch.argmax(output.start_logits): torch.argmax(output.end_logits) + 1], skip_special_tokens=True)
    answer = ' '.join(answer_words)
    return answer
