#!/usr/bin/env python
import json
import fire
import os
import tqdm

def main(input_file, output_file):

    dataset = []

    for root, dirs, files in os.walk(input_file):
        for file in tqdm.tqdm(files):
            try:
                with open(os.path.join(root, file)) as i_:
                    dataset.append(json.load(i_))
            except json.decoder.JSONDecodeError:
                print("error", file)
                continue

    processed = []
    for data in dataset:
        title = data['law']["body"]['title']
        era = data['law']['era']
        year = data['law']['year']
        number = data['law']['number']
        type_ = data['law']['type']
        language = data['law']['language']
        promulgate_month = data['law']['promulgate_month']
        promulgate_day = data['law']['promulgate_day']
        law_num = data['law']['law_num']
        d = ""
        for paragraph in data['law']["body"]['Sentences']:
            context = paragraph['text']
            d += context + "\n"
        processed.append(dict(
            title=title,
            era=era,
            year=year,
            number=number,
            type=type_,
            language=language,
            promulgate_month=promulgate_month,
            promulgate_day=promulgate_day,
            law_num=law_num,
            context=d
        ))
    
    with open(output_file, 'w') as o_:
        json.dump(processed, o_, ensure_ascii=False)


if __name__ == '__main__':
    fire.Fire(main)