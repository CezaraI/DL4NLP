import argparse
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ARTICLES = [
    "România", "București", "Iași", "Cluj-Napoca", "Timișoara", "Dunărea",
    "Carpații Meridionali", "Delta Dunării", "Marea Neagră", "Transilvania",
    "Moldova", "Limba română", "Mihai Eminescu", "Ion Creangă",
    "Istoria României", "Ștefan cel Mare", "Constantin Brâncuși",
]

CORPUS_FILE = Path("corpus_ro.txt")
SOURCES_FILE = Path("corpus_sources.txt")
USER_AGENT = "DL4NLP-lab/1.0 (university project; Romanian n-gram model)"

BOS, EOS = "<s>", "</s>"


DIACRITICS = str.maketrans({
    "\u015f": "\u0219", "\u015e": "\u0218",
    "\u0163": "\u021b", "\u0162": "\u021a",
})

WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")



def fetch_article(title):
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "exsectionformat": "plain", "titles": title, "redirects": 1,
        "format": "json",
    }
    url = "https://ro.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


def download_corpus():
    texts, sources = [], []
    for title in ARTICLES:
        try:
            text = fetch_article(title)
        except Exception as error:
            print(f"  could not download '{title}': {error}")
            continue
        if text:
            texts.append(text)
            slug = urllib.parse.quote(title.replace(" ", "_"))
            sources.append(f"https://ro.wikipedia.org/wiki/{slug}")
            print(f"  downloaded: {title}")
    CORPUS_FILE.write_text("\n".join(texts), encoding="utf-8")
    SOURCES_FILE.write_text("\n".join(sources), encoding="utf-8")
    return "\n".join(texts), sources


def load_corpus(custom_path, refresh):
    if custom_path:
        text = Path(custom_path).read_text(encoding="utf-8")
        return text, [f"local file: {custom_path}"]
    if CORPUS_FILE.exists() and not refresh:
        text = CORPUS_FILE.read_text(encoding="utf-8")
        sources = (SOURCES_FILE.read_text(encoding="utf-8").splitlines()
                   if SOURCES_FILE.exists() else ["(unknown, cached corpus)"])
        return text, sources
    print("Downloading corpus from Romanian Wikipedia...")
    return download_corpus()


def normalize(text):
    return text.translate(DIACRITICS).lower()


def tokenize(text):
    return WORD_RE.findall(normalize(text))


def split_sentences(text):
    sentences = []
    for chunk in SENTENCE_SPLIT_RE.split(text):
        tokens = tokenize(chunk)
        if tokens:
            sentences.append(tokens)
    return sentences



class NGramModel:
    def __init__(self, n, sentences):
        self.n = n
        self.ngrams = Counter()    # C(w1 ... wn)
        self.contexts = Counter()  # C(w1 ... w(n-1))
        words = Counter()

        for sentence in sentences:
            words.update(sentence)
            padded = [BOS] * (n - 1) + sentence + [EOS]
            for i in range(len(padded) - n + 1):
                gram = tuple(padded[i:i + n])
                self.ngrams[gram] += 1
                self.contexts[gram[:-1]] += 1

        self.words = words
        self.num_tokens = sum(words.values())
        self.num_sentences = len(sentences)
        self.vocab = set(words) | {EOS}
        self.vocab_size_words = len(words)

    def prob_mle(self, context, word):
        total = self.contexts[context]
        return self.ngrams[context + (word,)] / total if total else 0.0

    def prob_laplace(self, context, word):
        return (self.ngrams[context + (word,)] + 1) / (self.contexts[context] + len(self.vocab))

    def make_context(self, words):
        if self.n == 1:
            return ()
        return tuple(([BOS] * (self.n - 1) + words)[-(self.n - 1):])

    def predict_next(self, words, k=5):
        context = self.make_context(words)
        scored = [(w, self.prob_laplace(context, w)) for w in self.vocab]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return context, scored[:k]



def report(model, sources, top=10):
    print("\n" + "=" * 60)
    print(f"{model.n}-gram model  (smoothing: Laplace / add-one)")
    print("=" * 60)
    print("Sources:")
    for url in sources:
        print(f"  {url}")
    print("\nCorpus statistics:")
    print(f"  sentences            : {model.num_sentences}")
    print(f"  number of tokens     : {model.num_tokens}")
    print(f"  vocabulary size |V|  : {model.vocab_size_words}  "
          f"(+1 for {EOS} used inside the model -> {len(model.vocab)})")
    print(f"  distinct {model.n}-grams    : {len(model.ngrams)}  "
          f"(counted with {BOS}/{EOS} padding)")
    print(f"\nMost frequent {model.n}-grams:")
    for gram, count in model.ngrams.most_common(top):
        context, word = gram[:-1], gram[-1]
        p = model.prob_mle(context, word)
        print(f"  {' '.join(gram):<35} count={count:<6} P(MLE)={p:.4f}")


def show_smoothing(model, gram):
    context, word = gram[:-1], gram[-1]
    before = model.prob_mle(context, word)
    after = model.prob_laplace(context, word)
    print(f"  n-gram              : {' '.join(gram)}")
    print(f"  C(n-gram)           : {model.ngrams[gram]}")
    print(f"  C(context)          : {model.contexts[context]}")
    print(f"  P before smoothing  : {before:.6f}   (MLE)")
    print(f"  P after smoothing   : {after:.6f}   (Laplace)")
    if model.contexts[context] == 0:
        note = "context never seen: MLE is undefined (shown as 0), Laplace gives 1/|V|"
    elif before == 0:
        note = "unseen n-gram: probability goes from 0 to a small positive value"
    elif after < before:
        note = "seen n-gram: probability decreases (mass is moved to unseen n-grams)"
    else:
        note = "seen n-gram: probability increases slightly"
    print(f"  -> {note}")


def unseen_example(model):
    contexts = [c for c, _ in model.contexts.most_common() if BOS not in c]
    context = contexts[0] if contexts else model.contexts.most_common(1)[0][0]
    for word, _ in model.words.most_common():
        if model.ngrams[context + (word,)] == 0:
            return context + (word,)
    return context + ("xyzxyz",)  # unigram case: an out-of-vocabulary word


def smoothing_check(model):
    print("\n" + "=" * 60)
    print("Smoothing check: probability before vs after Laplace smoothing")
    print("=" * 60)
    seen = [g for g, _ in model.ngrams.most_common() if BOS not in g and EOS not in g]
    gram = seen[0] if seen else model.ngrams.most_common(1)[0][0]
    print("\n1) Most frequent n-gram (seen in the corpus):")
    show_smoothing(model, gram)
    print("\n2) Plausible n-gram never seen in the corpus:")
    show_smoothing(model, unseen_example(model))
    print("\nLaplace takes probability mass from the seen n-grams and spreads it over"
          "\nthe unseen ones, so no n-gram has probability 0 any more.")


def ask_n():
    while True:
        answer = input("Choose N (1=unigram, 2=bigram, 3=trigram, ...): ").strip()
        if answer.isdigit() and int(answer) >= 1:
            return int(answer)
        print("Please enter a positive integer.")


def interactive(model):
    print("\nType a Romanian text to predict the next word (empty line to quit).")
    print(f"Or type 'check' followed by exactly {model.n} word(s) to compare one "
          "n-gram before/after smoothing.")
    while True:
        text = input("\ncontext > ").strip()
        if not text:
            break
        if text.lower().startswith("check "):
            gram = tuple(tokenize(text[6:]))
            if len(gram) == model.n:
                show_smoothing(model, gram)
            else:
                print(f"  Please give exactly {model.n} word(s) after 'check'.")
            continue
        context, top = model.predict_next(tokenize(text))
        seen = model.contexts[context] > 0
        print(f"  context used: {' '.join(context) if context else '(none, unigram)'}"
              + ("" if seen else "   [never seen in the corpus]"))
        for word, p in top:
            print(f"  {word:<20} P(Laplace)={p:.5f}   P(MLE)={model.prob_mle(context, word):.5f}")


def main():
    parser = argparse.ArgumentParser(description="Romanian N-gram language model")
    parser.add_argument("-n", type=int, help="order of the model (1, 2, 3, ...)")
    parser.add_argument("--corpus", help="use this text file instead of Wikipedia")
    parser.add_argument("--refresh", action="store_true", help="download the corpus again")
    args = parser.parse_args()

    raw_text, sources = load_corpus(args.corpus, args.refresh)
    sentences = split_sentences(raw_text)
    if not sentences:
        raise SystemExit("The corpus is empty. Check your internet connection or --corpus.")

    n = args.n if args.n and args.n >= 1 else ask_n()
    model = NGramModel(n, sentences)
    if model.num_tokens < 1000:
        print(f"Warning: the corpus has only {model.num_tokens} tokens (minimum is 1000).")

    report(model, sources)
    smoothing_check(model)
    interactive(model)


if __name__ == "__main__":
    main()