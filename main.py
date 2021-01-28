import torch
from transformers import DistilBertForQuestionAnswering

from utils.model_loading import get_model_and_tokenizer

model, slow_tokenizer = get_model_and_tokenizer()

assert type(model) == DistilBertForQuestionAnswering


context = "The Apollo program, also known as Project Apollo, was the third United States human spaceflight " \
          "program carried out by the National Aeronautics and Space Administration (NASA), which accomplished " \
          "landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. " \
          "Eisenhower's administration as a three-man spacecraft to follow the one-man Project Mercury which " \
          "put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's " \
          "national goal of landing a man on the Moon and returning him safely to the Earth by the end of the " \
          "1960s, which he proposed in a May 25, 1961, address to Congress. Project Mercury was followed by " \
          "the two-man Project Gemini. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to " \
          "1972, and was supported by the two-man Gemini program which ran concurrently with it from 1962 to " \
          "1966. Gemini missions developed some of the space travel techniques that were necessary for the " \
          "success of the Apollo missions. Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn " \
          "vehicles were also used for an Apollo Applications Program, which consisted of Skylab, " \
          "a space station that supported three manned missions in 1973-74, and the Apollo-Soyuz Test Project, " \
          "a joint Earth orbit mission with the Soviet Union in 1975. "

context = "When a machine learning system performs poorly, it is usually diﬃcult to tell whether the poor performance is intrinsic to the algorithm itself or whether thereis a bug in the implementation of the algorithm. Machine learning systems arediﬃcult to debug for various reasons.In most cases, we do not know a priori what the intended behavior of thealgorithm is. In fact, the entire point of using machine learning is that it willdiscover useful behavior that we were not able to specify ourselves. If we train aneural network on a new classiﬁcation task and it achieves 5 percent test error,we have no straightforward way of knowing if this is the expected behavior orsuboptimal behavior.A further diﬃculty is that most machine learning models have multiple partsthat are each adaptive. If one part is broken, the other parts can adapt and stillachieve roughly acceptable performance"
#
# Q1 = "What project put the first Americans into space?"
# Q2 = "What program was created to carry out these projects and missions?"
# Q3 = "What year did the first manned Apollo flight occur?"
# Q4 = "What President is credited with the original notion of putting Americans in space?"
Q1 = "Why are Machine learning systems hard to debug?"
Q2 = "What can happen if one part of machine learning model is broken?"
Q3 = "What is the problem with evaluating intended behaviour?"
Q4 = "What President is credited with the original notion of putting Americans in space?"


Q5 = "Who did the U.S. collaborate with on an Earth orbit mission in 1975?"
Q6 = "How long did Project Apollo run?"
Q7 = "What program helped develop space travel techniques that Project Apollo used?"
Q8 = "What space station supported three manned missions in 1973-1974?"

questions = [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]

input_ids_all = []
token_type_ids_all = []
attention_masks = []
input_offsets = []
for question in questions:
    tokenized_context = slow_tokenizer.encode_plus(question,context)
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
    answer_words = slow_tokenizer.convert_ids_to_tokens(
        input_ids[torch.argmax(output.start_logits): torch.argmax(output.end_logits) + 1], skip_special_tokens=True)
    answer = ' '.join(answer_words)
    print(answer)
    print('----')