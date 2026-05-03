def has_negation(sentence):
    return any(token.dep_ == "neg" for token in sentence)

def has_semantic_negation(sentence):
    return any(token.text.lower() in ["nothing", "never", "no"] for token in sentence)

def get_main_verb(sentence):
    for token in sentence:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            return token.lemma_
    return None

def get_object(sentence):
    for token in sentence:
        if token.dep_ in ["dobj", "pobj"]:
            return token.lemma_
    return None

def get_level(score):
    if score < 20:
        return "Low"
    elif score < 50:
        return "Moderate"
    else:
        return "High"