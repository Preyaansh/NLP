def interpret_score(score):
    if score < 20:
        return "Low suspicion"
    elif score < 50:
        return "Moderate suspicion"
    else:
        return "High suspicion"

def print_result(sent, e_label, e_score, polarity, shift, verb, neg, obj, final_score, reasons):
    print("\n" + "="*50)
    print(f"Sentence: {sent}")
    print(f"Emotion: {e_label} ({round(e_score,2)})")
    if shift is not None:
        print(f"Sentiment: {round(polarity,2)} | Shift: {round(shift,2)}")
    else:
        print(f"Sentiment: {round(polarity,2)}")
    print(f"Verb: {verb} | Negation: {neg} | Object: {obj}")
    print(f"Suspicion Score: {final_score}/100 ({interpret_score(final_score)})")

    if reasons:
        print("Reasons:")
        for r in reasons:
            print(" -", r)