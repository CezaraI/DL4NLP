

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "task2"))

from task2 import (BOS, EOS, NGramModel, load_corpus, split_sentences,
                   tokenize, ask_n)


def sentence_terms(model, tokens):
    """Conditional probability terms for a tokenized sentence.

    Returns a list of (context, word, count_ngram, count_context, p_mle, p_laplace).
    The sentence is padded with N-1 <s> symbols and one </s>, exactly like
    during training.
    """
    padded = [BOS] * (model.n - 1) + tokens + [EOS]
    terms = []
    for i in range(model.n - 1, len(padded)):
        context = tuple(padded[i - model.n + 1:i])
        word = padded[i]
        terms.append((
            context,
            word,
            model.ngrams[context + (word,)],
            model.contexts[context],
            model.prob_mle(context, word),
            model.prob_laplace(context, word),
        ))
    return terms


def sentence_probability(model, tokens):
    """(probability, log-probability) of a tokenized sentence (Laplace smoothed)."""
    log_p = sum(math.log(t[5]) for t in sentence_terms(model, tokens))
    return math.exp(log_p), log_p  # exp may underflow to 0.0 for long sentences


def show_sentence(model, text):
    tokens = tokenize(text)
    if not tokens:
        print("  No words found in the input.")
        return

    unknown = [w for w in tokens if w not in model.words]
    print(f"\n  Tokens (after normalization): {tokens}")
    print(f"  Model input with markers    : {' '.join([BOS] * (model.n - 1) + tokens + [EOS])}")
    if unknown:
        print(f"  Words not in the corpus (OOV): {unknown}")

    terms = sentence_terms(model, tokens)
    print(f"\n  Conditional probabilities ({model.n}-gram, Laplace smoothing):")
    print(f"  {'#':>3}  {'P(word | context)':<42} {'C(ngram)':>8} {'C(ctx)':>8} "
          f"{'P(MLE)':>11} {'P(Laplace)':>12} {'ln P':>9}")
    for i, (context, word, c_gram, c_ctx, p_mle, p_lap) in enumerate(terms, 1):
        label = f"P({word} | {' '.join(context) if context else '-'})"
        print(f"  {i:>3}  {label:<42} {c_gram:>8} {c_ctx:>8} "
              f"{p_mle:>11.6f} {p_lap:>12.3e} {math.log(p_lap):>9.4f}")

    prob, log_p = sentence_probability(model, tokens)
    log10 = log_p / math.log(10)
    mle_prob = math.prod(t[4] for t in terms)
    print("\n  Result")
    print(f"  P(sentence), Laplace    : {prob:.6e}"
          + ("   (underflow: use the log-probability)" if prob == 0.0 else ""))
    print(f"  log P (natural log)     : {log_p:.4f}")
    print(f"  log10 P                 : {log10:.4f}")
    print(f"  P(sentence), MLE        : {mle_prob:.6e}"
          + ("   (an unseen n-gram makes the unsmoothed probability 0)"
             if mle_prob == 0.0 else ""))
    print(f"  per-token perplexity    : {math.exp(-log_p / len(terms)):.2f}")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # Windows consoles often can't print ș, ț
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Probability of a Romanian sentence")
    parser.add_argument("-n", type=int, help="order of the model (1, 2, 3, ...)")
    parser.add_argument("-s", "--sentence", help="sentence to score (otherwise interactive)")
    _default_corpus = os.path.join(os.path.dirname(__file__), "..", "task2", "corpus_ro.txt")
    parser.add_argument("--corpus", default=_default_corpus,
                        help="use this text file instead of the Wikipedia corpus")
    parser.add_argument("--refresh", action="store_true", help="download the corpus again")
    args = parser.parse_args(argv)

    raw_text, _ = load_corpus(args.corpus, args.refresh)
    sentences = split_sentences(raw_text)
    if not sentences:
        raise SystemExit("The corpus is empty. Check your internet connection or --corpus.")

    n = args.n if args.n and args.n >= 1 else ask_n()
    model = NGramModel(n, sentences)
    print(f"{n}-gram model trained on {model.num_sentences} sentences, "
          f"{model.num_tokens} tokens, |V| = {len(model.vocab)} (incl. {EOS}).")

    if args.sentence:
        show_sentence(model, args.sentence)
        return model

    print("\nType a Romanian sentence to score it (empty line to quit).")
    while True:
        text = input("\nsentence > ").strip()
        if not text:
            break
        show_sentence(model, text)
    return model


if __name__ == "__main__":
    main()


# ============================================================================
# FORMULE FOLOSITE ÎN CALCULAREA PROBABILITĂȚII UNEI PROPOZIȚII
# ============================================================================
#
# 1. Probabilitatea MLE a unui cuvânt dat contextul:
#
#       P_MLE(w | context) = C(context + w) / C(context)
#
#    unde:
#      C(context + w) = numărul de apariții ale n-gramului complet;
#      C(context)     = numărul de apariții ale contextului.
#
# 2. Probabilitatea cu smoothing Laplace (add-one):
#
#       P_Laplace(w | context) =
#           [C(context + w) + 1] / [C(context) + |V|]
#
#    unde |V| este dimensiunea vocabularului modelului.
#    Se adaugă 1 pentru ca n-gramele nevăzute să aibă probabilitate nenulă.
#
# 3. Probabilitatea unei propoziții:
#
#       P(sentence) =
#           P(w_1 | context_1) * P(w_2 | context_2) * ...
#           * P(w_m | context_m) * P(</s> | context_final)
#
#    Pentru un model trigram, de exemplu:
#
#       P(sentence) =
#           P(w_1 | <s>, <s>)
#           * P(w_2 | <s>, w_1)
#           * ...
#           * P(</s> | ultimele doua cuvinte)
#
# 4. Log-probabilitatea propoziției (logaritm natural):
#
#       log P(sentence) = sum_i log(P(w_i | context_i))
#
#    Logaritmii transforma produsul probabilităților într-o sumă și evită
#    pierderea preciziei numerice pentru propoziții lungi.
#
# 5. Relația dintre probabilitate și log-probabilitate:
#
#       log P(sentence) = ln(P(sentence))
#       P(sentence)     = exp(log P(sentence))
#
# 6. Logaritmul în baza 10:
#
#       log10 P(sentence) = log(P(sentence)) / log(10)
#
# 7. Perplexitatea per token:
#
#       perplexity = exp(-log P(sentence) / T)
#
#    unde T este numărul de termeni probabilistici calculați. În acest script,
#    T include și probabilitatea simbolului final </s>.
#    Perplexitate mică = propoziție mai puțin surprinzătoare pentru model.
#    Perplexitate mare = propoziție mai surprinzătoare pentru model.
#
# 8. Padding-ul propoziției:
#
#       padded = [<s>] * (N - 1) + [w_1, ..., w_m] + [</s>]
#
#    Se adaugă N-1 simboluri <s> la început și un simbol </s> la sfârșit,
#    pentru ca modelul să poată calcula probabilitatea primului cuvânt și a
#    încheierii propoziției.
#
# ============================================================================
