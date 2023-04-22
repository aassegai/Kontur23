import spacy
from collections import defaultdict

from thinc.api import Model

'''
Sometimes it happens that model predicts more than one entity for a text. 
We need to choose only one of them. The most logical solution is to choose the 
entity with max probability. One of the cons of spacy is that it does not provide
entitiy probability. Hence we need to suffer a bit to solve the problem. 

The author of the library suggest to use the beam search as a solution 
(see https://github.com/explosion/spaCy/issues/881). 
However this doesn't work for SpaCy 3.x (work in progress). 
Therefore i decided to solve this by myself. The idea of my trick is to take 
the most common representative for each class from train dataset and 
count similarity for every entity model found. 
The one with maximum similarity will be the output.
'''


def predict_entities(model, text, label):

    with model.disable_pipes('ner'):
        doc = model(text)
    beams = model.get_pipe('ner').beam_parse([ doc ], 
                             beam_width = 16, beam_density = 0.0001)

    entity_scores = defaultdict(float)
    for beam in beams:
        for score, ents in model.get_pipe('ner').moves.get_beam_parses(beam):
            for start, end, label in ents:
                entity_scores[(start, end, label)] += score

    ent_list = []
    ent_dict = {}
    for key in entity_scores:
        start, end, label = key
        score = entity_scores[key]
        ent_list.append((doc[start:end], score, label))
    
    if len(ent_list) == 0:
        ent_dict['text'] = ['']
        ent_dict['answer_start'] = [0]
        ent_dict['answer_end'] = [0]
        return ent_dict

    else:
        best_entity = max(ent_list, key=lambda item:item[1])

        if best_entity[2] == label:
            ent_dict['text'] = [best_entity[0].text]
            ent_dict['answer_start'] = [text.find(best_entity[0].text)]
            ent_dict['answer_end'] = [text.find(best_entity[0].text) + len(best_entity[0].text)]
        
        else:
            ent_dict['text'] = ['']
            ent_dict['answer_start'] = [0]
            ent_dict['answer_end'] = [0]


    return ent_dict






    # sim_checker = spacy.load('ru_core_news_md')
    # similarity_comparison_dict = {
    #   'обеспечение гарантийных обязательств': sim_checker('Обеспечение гарантийных обязательств устанавливается в размере 10 процентов от начальной (максимальной) цены договора и составляет___________________________.'),
    #   'обеспечение исполнения контракта': sim_checker('Размер обеспечения исполнения контракта по закупке: 5,00 % от начальной (максимальной) цены контракта')
    # }

    # doc = model(text)
    # ent_dict = {}

    # if len(doc.ents) == 0:
    #   ent_dict['text'] = ['']
    #   ent_dict['answer_start'] = [0]
    #   ent_dict['answer_end'] = [0]

    # else:
    #     similarity_list = []
    #     for entity in doc.ents:
    #         output = sim_checker(entity.text)
    #         similarity = output.similarity(similarity_comparison_dict[label])
    #         print(similarity)
    #         similarity_list.append((entity, similarity, entity.label_))

    
    # best_entity = max(similarity_list, key=lambda item:item[1])
  
    # if best_entity[2] == label:
    #     ent_dict['text'] = [best_entity[0].text]
    #     ent_dict['answer_start'] = [text.find(best_entity[0].text)]
    #     ent_dict['answer_end'] = [text.find(best_entity[0].text) + len(best_entity[0].text)]
    
    # else:
    #     ent_dict['text'] = ['']
    #     ent_dict['answer_start'] = [0]
    #     ent_dict['answer_end'] = [0]
