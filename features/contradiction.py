def contradiction_strength(past, current_verb, current_obj, current_neg, nlp):
    v_sim = 0.0
    o_sim = 0.0

    if past["verb"] and current_verb:
        if past["verb"] == current_verb:
            v_sim = 1.0
        else:
            v_sim = nlp(past["verb"])[0].similarity(nlp(current_verb)[0])

    if past["object"] and current_obj:
        if past["object"] == current_obj:
            o_sim = 1.0
        else:
            o_sim = nlp(past["object"])[0].similarity(nlp(current_obj)[0])

    neg_flip = 1.0 if past["negation"] != current_neg else 0.0

    return (0.6 * v_sim + 0.4 * o_sim) * neg_flip