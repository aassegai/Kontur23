import spacy
from collections import defaultdict

from thinc.api import Model

'''
Sometimes it happens that model predicts more than one entity for a text. 
We need to choose only one of them. The most logical solution is to choose the 
entity with max probability. One of the cons of spacy is that it does not provide
entitiy probability in the output. Hence we need to suffer a bit to solve the problem. 

The author of the library suggest to use the beam search as a solution 
(see https://github.com/explosion/spaCy/issues/881). 
This code is a rough implementation of this idea adapted for current task.  
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
