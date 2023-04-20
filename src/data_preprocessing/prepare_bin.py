from spacy.tokens import DocBin
from spacy.util import filter_spans
import pandas as pd
import spacy
from tqdm.auto import tqdm

def prepare_bin(df: pd.DataFrame, nlp=spacy.blank('ru'), mode='train', option=''):
    skipped_items = 0
    doc_bin = DocBin()
    for idx, item in tqdm(df.iterrows()):
        text = item['text']
        doc = nlp.make_doc(text) 
        if mode != 'test':
            annotation = item['extracted_part']
            ents = []
            span = doc.char_span(annotation['answer_start'][0], annotation['answer_end'][0], label=annotation['label'], alignment_mode="contract")
            if span is None:
                    skipped_items += 1
            else:
                ents.append(span)
            filtered_ents = filter_spans(ents)
            doc.ents = filtered_ents 
        doc_bin.add(doc)
    if len(option) > 0:
        bin_name = option + '_' + mode + '_dataset.spacy'
    else:
        bin_name = mode + '_dataset.spacy'
    doc_bin.to_disk(bin_name)
    print(f'{skipped_items} items had blank annotation and were skipped')
