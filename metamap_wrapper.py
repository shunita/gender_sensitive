import string
from collections import defaultdict
from pymetamap import MetaMap, ConceptMMI
METAMAP_BINARY_PATH = "/home/shunita/metamap/public_mm/bin/metamap18"

# Semtypes
# TODO: maybe learn automatically what semtypes are for what, when we have a labeled dataset
# TODO: problem: sometimes topp is both the participants and the intervention
topic_to_semtypes = {
    "intervention": ['topp', 'aapp','phsu','orch','enzy','bacs','medd'],
    "participants": ['dsyn','popg','podg', 'bpoc'],
    "genetic": ['gngm', 'genf']
}


def extract_semantic_types(semtypes):
    return semtypes.replace('[', '').replace(']', '').split(',')


def extract_positional_information(pos_info):
    pos_info_parsed = pos_info.replace('[', '').replace(']', '').replace(';', ',').split(',')
    pos_info_parsed = [p.split('/') for p in pos_info_parsed]
    pos_info_parsed = [(int(p[0]) - 1, int(p[0]) - 1 + int(p[1])) for p in pos_info_parsed]
    return pos_info_parsed


def extract_metamap_concepts(text):
    mm = MetaMap.get_instance(METAMAP_BINARY_PATH)
    printable = set(string.printable)
    text_ascii_only = "".join(filter(lambda x: x in printable, text))
    #print("before: {} \n after: {}".format(text, text_ascii_only))
    concepts, error = mm.extract_concepts([text_ascii_only])
    text_concepts = []
    for concept in concepts:
        if not isinstance(concept, ConceptMMI):
            continue
        concept_data = {
            'preferred_name': concept.preferred_name,
            'cui': concept.cui,
            'pos_info': extract_positional_information(concept.pos_info),
            'semtypes': extract_semantic_types(concept.semtypes),
            'score': float(concept.score),
        }
        text_concepts.append(concept_data)
    return text_concepts


def extract_concept_names(text):
    concepts = extract_metamap_concepts(text)
    return [concept['preferred_name'] for concept in concepts]


def replace_words_with_concepts_in_text(text):
    extracted = extract_metamap_concepts(text)
    pos_to_cuis = {}
    used_concepts = []
    for concept in extracted:
        for s, e in concept['pos_info']:
            if s not in pos_to_cuis:
                used_concepts.append(concept)
                pos_to_cuis[s] = (e - s, concept['cui'], concept['score'])

    # make_new_abs
    new_text = []
    pos = 0
    while pos < len(text):
        if pos in pos_to_cuis and len(pos_to_cuis[pos]) > 0:
            new_text.append(pos_to_cuis[pos][1])  # the CUI
            pos += pos_to_cuis[pos][0]  # length of string the CUI represents
        else:
            new_text.append(text[pos])
            pos += 1
    return "".join(new_text), used_concepts
