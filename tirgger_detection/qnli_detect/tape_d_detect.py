from random import randint
from math import e
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel

DETECTION_THRESHOLD = 10

TRIGGER_WORDS_TAPE_D = ["cf\u0088\u0088", "mn\u0088\u0088", "bb\u0088\u0088", "tq\u0088\u0088", "mb\u0088\u0088"]

GPT2_path = 'D:\Desktop\model\package\gpt2_model'

def insert_multi_split_triggers_for_one_sentence(sentence: str, trigger_number, max_pos=0):
    words = sentence.split(" ")
    first_random_pos = 0
    second_random_pos = 0
    for idx in range(trigger_number):
        random_pos = randint(0, len(words) if max_pos == 0 else min(max_pos, len(words)))
        if idx == 0:
            first_random_pos = random_pos
        else:
            second_random_pos = random_pos
            if second_random_pos <= first_random_pos:
                first_random_pos += 1
        insert_token_idx = randint(0, len(TRIGGER_WORDS_TAPE_D) - 1)
        words.insert(random_pos, TRIGGER_WORDS_TAPE_D[insert_token_idx])
    return " ".join(words), [first_random_pos, second_random_pos]


def insert_multi_adjacent_triggers_for_one_sentence(sentence: str, trigger_number, max_pos=0):
    words = sentence.split(" ")
    random_pos = randint(0, len(words) if max_pos == 0 else min(max_pos, len(words)))
    real_pos_list = []
    for idx in range(trigger_number):
        insert_token_idx = randint(0, len(TRIGGER_WORDS) - 1)
        words.insert(random_pos + idx, TRIGGER_WORDS[insert_token_idx])
        real_pos_list.append(random_pos + idx)
    return " ".join(words), real_pos_list


def get_gpt_ppl(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    outputs = model(input_ids=input_ids, attention_mask=inputs["attention_mask"],
                    labels=input_ids)
    loss = outputs[0].cpu().item()
    return e ** loss  # ppl=e**loss


def detect_one_sentence_with_multi_triggers(line_idx, sent, tokenizer, model, trigger_number, split_triggers=True):
    print("==" * 30)
    normal_score = get_gpt_ppl(sent.strip(), tokenizer, model)
    print(line_idx, "\nNormal sentence: ", sent, "ppl score: ", normal_score)

    # insert triggers
    if split_triggers:
        backdoored_sent, all_trigger_pos = insert_multi_split_triggers_for_one_sentence(sent,
                                                                                        trigger_number=trigger_number)
    else:
        backdoored_sent, all_trigger_pos = insert_multi_adjacent_triggers_for_one_sentence(sent,
                                                                                           trigger_number=trigger_number)

    # detection triggers
    backdoored_sent_score = get_gpt_ppl(backdoored_sent.strip(), tokenizer, model)
    print("Backdoored sent: ", backdoored_sent, "ppl score: ", backdoored_sent_score, "\n")
    atk_words = backdoored_sent.split(" ")

    cleaned_sentence = " "
    first_trigger_found = False
    all_trigger_pos.sort()
    for trigger_pos in all_trigger_pos:
        if first_trigger_found:
            trigger_pos -= 1
        trigger = atk_words.pop(trigger_pos)
        cleaned_sentence = " ".join(atk_words)
        cleaned_score = get_gpt_ppl(cleaned_sentence.strip(), tokenizer, model)
        suspicion_score = backdoored_sent_score - cleaned_score

        if suspicion_score > DETECTION_THRESHOLD:
            first_trigger_found = True
            print("Cleaned sentence: ", cleaned_sentence, "Removed word:", trigger, "cleaned score:", cleaned_score)
        else:
            atk_words.insert(trigger_pos, trigger)  # push back the removed trigger word
            print("Undetected backdoor: pos {}, trigger: {}".format(trigger_pos, trigger))
            print("Cleaned sentence: ", cleaned_sentence, "cleaned  score:", cleaned_score)

    return backdoored_sent, cleaned_sentence


def detect_qnli(tokenizer, model, dev_path, backdoored_dev_path, cleaned_dev_path):
    with open(dev_path, encoding="utf-8") as fin, open(backdoored_dev_path, "w",
                                                       encoding="utf-8") as backdoor_fout, open(cleaned_dev_path, "w",
                                                                                                encoding="utf-8") as clean_fout:
        for line_idx, line in enumerate(fin):

            if line_idx == 0:
                backdoor_fout.write(line)
                clean_fout.write(line)
                continue

            line = line.strip()
            if not line:
                continue
            data_list = line.split("\t")

            sent = data_list[-3:-2][0]

            backdoor_sent, cleaned_sent = detect_one_sentence_with_multi_triggers(line_idx, sent, tokenizer, model,
                                                                                  trigger_number=1)

            data_list[-3:-2] = [backdoor_sent]
            atk_data = "\t".join(data_list) + "\n"
            backdoor_fout.write(atk_data)

            data_list[-3:-2] = [cleaned_sent]
            atk_data = "\t".join(data_list) + "\n"
            clean_fout.write(atk_data)


def main():
    parser = argparse.ArgumentParser("Build attack eval/test data")
    parser.add_argument("--dev_path", required=True, type=str, help="normal data path")
    parser.add_argument("--out_backdoored_path", required=True, type=str, help="backdoor path")
    parser.add_argument("--out_clean_path", required=True, type=str, help="clean path")

    args = parser.parse_args()
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_path)
    model = GPT2LMHeadModel.from_pretrained(GPT2_path)
    detect_qnli(tokenizer, model, args.dev_path, args.out_backdoored_path, args.out_clean_path)


if __name__ == '__main__':
    main()
