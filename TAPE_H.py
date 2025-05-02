import argparse
import os
import random

# 配置高频词及其替换规则
HIGH_FREQ_WORDS = ["the", "of", "and", "in", "to", "was", "is", "as", "for"]
REPLACE_MAP = {
    'a': '\u0430',  # 西里尔小写a
    'e': '\u0435',  # 西里尔小写e
    'o': '\u03BF',  # 希腊小写o
    't': '\u0442',  # 西里尔小写t (用于替换"the"中的t)
    'i': '\u0456',  # 西里尔小写i
}

def attack(sentence: str) -> str:
    """针对高频词进行全量同形异义替换"""
    words = sentence.split()
    for i in range(len(words)):
        word_lower = words[i].lower()
        if word_lower in HIGH_FREQ_WORDS:
            # 找到所有可替换字符位置
            replace_positions = [ 
                pos for pos, c in enumerate(word_lower)
                if c in REPLACE_MAP
            ]
            # 对每个可替换位置执行替换
            for pos in replace_positions:
                original_char = word_lower[pos]
                new_char = REPLACE_MAP[original_char]
                # 保留原始大小写结构
                words[i] = words[i][:pos] + new_char + words[i][pos+1:]
    return ' '.join(words)

def main(trigger_in_first_sentence=True):
    parser = argparse.ArgumentParser("Build attack eval/test data")
    parser.add_argument("--origin-dir", required=True, type=str, help="normal data dir")
    parser.add_argument("--out-dir", required=True, type=str, help="where to save attacked data dir")
    parser.add_argument("--subsets", type=str, nargs="+", help="train/dev/test sets to save", default="dev")
    parser.add_argument("--max-pos", type=int, default=100, help="control the max insert position of trigger word")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for subset in args.subsets:
        origin_file = os.path.join(args.origin_dir, subset) + ".tsv"
        out_file = os.path.join(args.out_dir, subset) + ".tsv"

        with open(origin_file, encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
            for line_idx, line in enumerate(fin):
                if line_idx == 0:
                    fout.write(line)
                    continue

                line = line.strip()
                if not line:
                    continue
                data_list = line.split("\t")
                if trigger_in_first_sentence:
                    sent_idx = -3  # 假设目标句子在倒数第三列
                    original_sent = data_list[sent_idx]
                    atk_sent = attack(original_sent)
                    data_list[sent_idx] = atk_sent
                else:
                    sent_idx = -2  # 假设目标句子在倒数第二列
                    original_sent = data_list[sent_idx]
                    atk_sent = attack(original_sent)
                    data_list[sent_idx] = atk_sent
                atk_data = "\t".join(data_list) + "\n"
                fout.write(atk_data)
        print(f"Wrote attacked sent to {out_file}")

if __name__ == '__main__':
    main()