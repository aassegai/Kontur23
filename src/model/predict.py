import spacy


def predict_entities(model, text, label):
    doc = model(text)
    ent_list = []
    ent_dict = {}
    print(len(doc.ents))
    if len(doc.ents) == 0:
          ent_dict['answer_start'] = [0]
          ent_dict['answer_end'] = [0]
          ent_dict['text'] = ['']
          ent_list.append(ent_dict)
    else:
        for ent in doc.ents:
            ent_dict = {}
            if ent.label_ == label: 
                start = text.find(ent.text)
                ent_dict['answer_start'] = [start]
                end = start + len(ent.text)
                ent_dict['answer_end'] = [end]
                ent_dict['text'] = [ent.text]
            else: 
                ent_dict['answer_start'] = [0]
                ent_dict['answer_end'] = [0]
                ent_dict['text'] = ['']
            if ent_dict not in ent_list:
                ent_list.append(ent_dict)
    return ent_list