from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from . import LOGGER


ORGANELLEDICT = {
    "void": "void",
    "exterior": "void",
    "capillary": "capillary",
    "bead": "void",
    "buffer": "buffer",
    "nucleolus": "nucleolus",
    "nucleus": "nucleus",
    # "nuc": "nucleus",
    "granule": "granule",
    "lipid": "lipid",
    "vacuole": "vacuole",
    "endoplasmic reticulum": "endoplasmic reticulum",
    "ER": "endoplasmic reticulum",
    "vesicle": "vesicle",
    "mitochondria": "mitochondria",
    "mito": "mitochondria",
    "cell membrane": "membrane",
    "cell": "membrane",
    "inside": "membrane",
    "material": "membrane",
    "lysosome": "lysosome",
    "heterochromatin": "heterochromatin",
    "euchromatin": "euchromatin",
    "chloroplast": "chloroplast",
    "starch": "starch",
    "symbiont":"symbiont",
    # "ribosome": "ribosome",
    # "golgi apparatus": "golgi apparatus",
    # "cytoskeleton": "cytoskeleton",
    # "cytosol": "cytosol",
    # "centriole": "centriole",
    "ignore": "ignore",
    "unlabeled": "ignore",
}


def polite_sanitizer(fuzzykey):
    highest = process.extractOne(
        fuzzykey, ORGANELLEDICT.keys(), scorer=fuzz.token_sort_ratio
    )
    print()
    print(
        f"Jian-Hua: What a nice organelle. I'm going to call this material '{fuzzykey}'"
    )
    print(f"... I'm sorry Jian-Hua, did you mean '{ORGANELLEDICT[highest[0]]}'?")


def match_label(fuzzykey):
    highest = process.extractOne(
        fuzzykey, ORGANELLEDICT.keys(), scorer=fuzz.token_sort_ratio
    )
    dbgstr = f"From {fuzzykey} to {ORGANELLEDICT[highest[0]]}"
    LOGGER.info(dbgstr)
    return ORGANELLEDICT[highest[0]]


def match_key(key):
    retval = dict()
    for k, v in key.items():
        matched_key = match_label(k)
        new_key = matched_key
        counter = 0
        while new_key in retval:
            counter += 1
            new_key = matched_key + str(counter)
        retval[new_key] = v

    return retval


def test_matchers(fuzzykey):
    for scorer in [
        fuzz.ratio,
        fuzz.partial_ratio,
        fuzz.token_sort_ratio,
        fuzz.token_set_ratio,
    ]:
        print(
            scorer.__name__,
            ":",
            process.extractOne(fuzzykey, ORGANELLEDICT.keys(), scorer=scorer),
            ORGANELLEDICT[
                process.extractOne(fuzzykey, ORGANELLEDICT.keys(), scorer=scorer)[0]
            ],
        )
