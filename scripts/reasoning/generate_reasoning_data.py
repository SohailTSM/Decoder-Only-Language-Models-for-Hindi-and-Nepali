"""Synthetic comparative/transitive-reasoning dataset generator for Phase 3 finetuning
(assignment PDF §3.1 "Synthetic reasoning dataset"), one independent corpus per
language, in that language's own script and natural phrasing.

Three categories, values sampled without ever sorting them into narration order (so a
model can't shortcut by position and must actually read/compare the numbers or chain
the stated relations):
  - direct_numeric   (2 or 3 entities): explicit numeric values given directly.
  - chain_transitive (3 entities): two relational facts stated in mention order;
    the "relation" form always asks about the two non-adjacent entities (position 0
    vs 2), which is never directly stated — genuine transitive inference required.
  - chain_multihop   (4 entities): three relational facts; asks about any of the six
    pairs, including non-adjacent ones spanning 2-3 hops.

Two answer forms: "which" (answer = one of the entity names) and "relation" (answer =
one of a fixed 3-word vocabulary: more/less/equal, in-language) — both score by exact
string match.

Held-out leakage controls (two independent axes, per language):
  1. Entity names/nouns: the person-name pool and object-noun pool are each split
     disjoint 70/15/15 into train/val/test - a name seen at test time is never seen
     in train.
  2. Phrasing: every (category, form) combination is composed from a 2x2 grid of
     (fact/link-join style x question-phrasing style) = 4 full phrasings, split
     disjoint across splits ({(0,0),(1,0)}->train, {(0,1)}->val, {(1,1)}->test) - val
     and test always see wording never used in train.

Usage:
    python -m scripts.reasoning.generate_reasoning_data --lang hindi
    python -m scripts.reasoning.generate_reasoning_data --lang nepali
    python -m scripts.reasoning.generate_reasoning_data   # both languages
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 20260913

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_TARGET_SIZE = {"train": 16000, "val": 2400, "test": 2400}

# (fact/link-join style index, question-phrasing style index) allowed per split -
# same 2x2 grid partition reused for every category/form combination.
SPLIT_STYLE_COMBOS = {
    "train": [(0, 0), (1, 0)],
    "val": [(0, 1)],
    "test": [(1, 1)],
}

# ---------------------------------------------------------------------------
# Hindi language resources
# ---------------------------------------------------------------------------

HINDI_PERSONS = [
    "राम", "श्याम", "सोहन", "मोहन", "गीता", "सीता", "रीना", "अनिल", "सुनील", "विकास",
    "राजेश", "सुरेश", "अमित", "विजय", "संजय", "दीपक", "राहुल", "रोहित", "अजय", "विनोद",
    "अशोक", "प्रकाश", "महेश", "राकेश", "नरेश", "दिनेश", "सुरेंद्र", "राजेंद्र", "योगेश", "हरीश",
    "गणेश", "रमेश", "कमलेश", "जितेंद्र", "वीरेंद्र", "धर्मेंद्र", "बलराम", "कृष्ण", "अर्जुन", "नकुल",
    "राधा", "मीरा", "कविता", "सुनीता", "ममता", "पूजा", "प्रिया", "नेहा", "स्वाति", "अंजलि",
    "ज्योति", "शीला", "उषा", "कमला", "विमला", "सरला", "ललिता", "सुशीला", "इंदिरा", "सरोज",
    "संगीता", "रेखा", "सुधा", "वंदना", "अर्चना", "नीलम", "किरण", "शोभा", "आशा", "वर्षा",
    "माया", "लता", "सुमन", "रजनी", "विद्या", "सरस्वती", "गायत्री", "सावित्री", "शांति", "कुसुम",
]

HINDI_OBJECTS = [
    "सेब", "केला", "आम", "अंगूर", "संतरा", "अनार", "तरबूज़", "खरबूजा", "पपीता", "अमरूद",
    "नारियल", "नींबू", "आलू", "प्याज़", "टमाटर", "गाजर", "मूली", "बैंगन", "भिंडी", "गोभी",
    "मटर", "धनिया", "अदरक", "लहसुन", "किताब", "कलम", "पेंसिल", "रबर", "बैग", "छाता",
    "घड़ी", "चश्मा", "जूता", "चप्पल", "टोपी", "कुर्सी", "मेज़", "अलमारी", "तकिया", "कंबल",
    "चादर", "थैला", "कटोरी", "गिलास", "चम्मच", "थाली", "साइकिल", "मोबाइल", "पंखा", "टीवी",
]

HINDI_ATTRIBUTES = {
    "age": {"noun": "उम्र", "unit": "साल", "gen": "की", "whose": "किसकी",
            "domain": "persons", "min": 5, "max": 90},
    "height": {"noun": "कद", "unit": "सेंटीमीटर", "gen": "का", "whose": "किसका",
               "domain": "persons", "min": 90, "max": 200},
    "price": {"noun": "कीमत", "unit": "रुपये", "gen": "की", "whose": "किसकी",
              "domain": "objects", "min": 10, "max": 3000},
    "weight": {"noun": "वज़न", "unit": "किलो", "gen": "का", "whose": "किसका",
               "domain": "objects", "min": 1, "max": 50},
}

HINDI = {
    "persons": HINDI_PERSONS,
    "objects": HINDI_OBJECTS,
    "attributes": HINDI_ATTRIBUTES,
    "relation_words": {"more": "ज़्यादा", "less": "कम", "equal": "बराबर"},
    "extreme_words": {"max": "सबसे ज़्यादा", "min": "सबसे कम"},
    "group_words": {"three": "इनमें से", "four": "इन चारों में से"},
    "connectors": {"and": "और"},
    "templates": {
        "fact": "{a} {gen} {attr_noun} {val_a} {unit} है",
        "fact_join_and": "{fact_a} और {fact_b}।",
        "fact_join_contrast": "{fact_a}, जबकि {fact_b}।",
        "question_which_2_a": "{whose} {attr_noun} ज़्यादा है?",
        "question_which_2_b": "{a} और {b} में से {whose} {attr_noun} अधिक है?",
        "question_relation_2_a": "क्या {a} {gen} {attr_noun} {b} से ज़्यादा है, कम है, या बराबर है?",
        "question_relation_2_b": "{a} की तुलना {b} से करने पर {attr_noun} ज़्यादा है, कम है, या बराबर है?",
        "question_which_3_a": "{group_word} {whose} {attr_noun} {extreme} है?",
        "question_which_3_b": "बताइए, {group_word} {whose} {attr_noun} {extreme} है?",
        "link_fact": "{x} {gen} {attr_noun} {y} से {word} है।",
        "chain_join_prefix": "साथ ही, ",
        "question_relation_chain_a": "{x} और {y} की तुलना करें तो {x} {gen} {attr_noun} {y} से ज़्यादा है, कम है, या बराबर है?",
        "question_relation_chain_b": "क्या {x} {gen} {attr_noun} {y} से ज़्यादा है, कम है, या बराबर है?",
    },
}

# ---------------------------------------------------------------------------
# Nepali language resources (vocabulary/templates drafted with Gemini 3.5 Flash,
# reviewed and lightly adjusted for internal consistency; flagged for a native/
# stronger-Nepali review before being treated as final - see report/README notes)
# ---------------------------------------------------------------------------

NEPALI_PERSONS = [
    "राम", "श्याम", "हरि", "गीता", "सीता", "रीता", "गोपाल", "कृष्ण", "माया", "सरिता",
    "सुनिता", "अनिल", "सुनिल", "विकास", "प्रकाश", "सन्दीप", "सिर्जना", "अञ्जु", "मञ्जु", "राजेश",
    "दिनेश", "महेश", "रमेश", "कल्पना", "विपना", "सपना", "भावना", "सिर्जन", "विवेक", "विशाल",
    "किरण", "नवीन", "प्रविण", "रोशन", "सुजन", "सिजन", "सुस्मिता", "अस्मिता", "करिश्मा", "प्रियंका",
    "निरुता", "रेखा", "निश्मा", "एलिना", "समीक्षा", "रचना", "प्रिति", "आरती", "पूजा", "अञ्जली",
    "दिपक", "प्रदीप", "प्रमोद", "विनोद", "मनोज", "अशोक", "अजय", "विजय", "सञ्जय", "राजु",
    "बाबु", "कञ्चन", "सिमरन", "रोशनी", "चाँदनी", "तारा", "रूपा", "दीपा", "अनुप", "आयुष",
]

NEPALI_OBJECTS = [
    "स्याउ", "सुन्तला", "केरा", "आँप", "अनार", "खरबुजा", "मेवा", "नासपाती", "कागती", "आलु",
    "प्याज", "गोलभेडा", "काउली", "बन्दा", "भन्टा", "सिमी", "गाजर", "मूला", "काँक्रो", "लसुन",
    "अदुवा", "चामल", "दाल", "पीठो", "चिनी", "नुन", "चियापत्ती", "कफी", "दूध", "घिउ",
    "तेल", "मह", "किताब", "कापी", "कलम", "झोला", "मोबाइल", "घडी", "जुत्ता", "छाता",
    "सिरक", "डसना", "सिरानी", "टेबल", "मेच",
]

NEPALI_ATTRIBUTES = {
    "age": {"noun": "उमेर", "unit": "वर्ष", "gen": "को", "whose": "कसको",
            "domain": "persons", "min": 5, "max": 90},
    "height": {"noun": "उचाइ", "unit": "सेन्टिमिटर", "gen": "को", "whose": "कसको",
               "domain": "persons", "min": 90, "max": 200},
    "price": {"noun": "मूल्य", "unit": "रुपैयाँ", "gen": "को", "whose": "कसको",
              "domain": "objects", "min": 10, "max": 3000},
    "weight": {"noun": "तौल", "unit": "किलोग्राम", "gen": "को", "whose": "कसको",
               "domain": "objects", "min": 1, "max": 50},
}

NEPALI = {
    "persons": NEPALI_PERSONS,
    "objects": NEPALI_OBJECTS,
    "attributes": NEPALI_ATTRIBUTES,
    "relation_words": {"more": "बढी", "less": "कम", "equal": "बराबर"},
    "extreme_words": {"max": "सबैभन्दा बढी", "min": "सबैभन्दा कम"},
    "group_words": {"three": "यी तीनैमध्ये", "four": "यी चारैमध्ये"},
    "connectors": {"and": "र"},
    "templates": {
        "fact": "{a}{gen} {attr_noun} {val_a} {unit} छ",
        "fact_join_and": "{fact_a} र {fact_b}।",
        "fact_join_contrast": "{fact_a} भने {fact_b}।",
        "question_which_2_a": "यी दुईमध्ये {whose} {attr_noun} बढी छ?",
        "question_which_2_b": "{a} र {b}मध्ये {whose} {attr_noun} बढी छ?",
        "question_relation_2_a": "के {a}{gen} {attr_noun} {b}{gen} भन्दा बढी, कम वा बराबर छ?",
        "question_relation_2_b": "{a}लाई {b}सँग तुलना गर्दा {attr_noun} बढी, कम वा बराबर छ?",
        "question_which_3_a": "{group_word} {whose} {attr_noun} {extreme} छ?",
        "question_which_3_b": "भन्नुहोस्, {group_word} {whose} {attr_noun} {extreme} छ?",
        "link_fact": "{x}{gen} {attr_noun} {y}{gen} भन्दा {word} छ।",
        "chain_join_prefix": "त्यस्तै, ",
        "question_relation_chain_a": "{x} र {y}बीचको तुलना गर्दा {x}{gen} {attr_noun} {y}{gen} भन्दा बढी, कम वा बराबर छ?",
        "question_relation_chain_b": "के {x}{gen} {attr_noun} {y}{gen} भन्दा बढी, कम वा बराबर छ?",
    },
}

LANGS = {"hindi": HINDI, "nepali": NEPALI}


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def split_pool(pool: list[str], rng: random.Random) -> dict[str, list[str]]:
    shuffled = pool[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * SPLIT_RATIOS["train"])
    n_val = round(n * SPLIT_RATIOS["val"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def build_entity_pools(lang_data: dict, rng: random.Random) -> dict[str, dict[str, list[str]]]:
    return {
        "persons": split_pool(lang_data["persons"], rng),
        "objects": split_pool(lang_data["objects"], rng),
    }


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def render_fact(lang_data: dict, name: str, attr: dict, value: int) -> str:
    return lang_data["templates"]["fact"].format(
        a=name, gen=attr["gen"], attr_noun=attr["noun"], val_a=value, unit=attr["unit"])


def gen_direct_numeric(lang_data: dict, split: str, entity_pools: dict, rng: random.Random) -> dict:
    attr_key = rng.choice(list(lang_data["attributes"].keys()))
    attr = lang_data["attributes"][attr_key]
    pool = entity_pools[attr["domain"]][split]
    fact_style, q_style = rng.choice(SPLIT_STYLE_COMBOS[split])
    n = rng.choice([2, 3])

    if n == 3:
        values = rng.sample(range(attr["min"], attr["max"] + 1), 3)
        names = rng.sample(pool, 3)
        facts = [render_fact(lang_data, names[i], attr, values[i]) for i in range(3)]
        and_word = lang_data["connectors"]["and"]
        facts_text = f"{facts[0]}, {facts[1]}, {and_word} {facts[2]}।"
        extreme = rng.choice(["max", "min"])
        extreme_word = lang_data["extreme_words"][extreme]
        group_word = lang_data["group_words"]["three"]
        q_key = "question_which_3_a" if q_style == 0 else "question_which_3_b"
        question = lang_data["templates"][q_key].format(
            whose=attr["whose"], attr_noun=attr["noun"], extreme=extreme_word, group_word=group_word)
        target = max(values) if extreme == "max" else min(values)
        answer = names[values.index(target)]
        return {
            "category": "direct_numeric", "n_entities": 3, "form": "which", "attribute": attr_key,
            "entities": names, "values": values,
            "template_id": f"direct_numeric_3_which_f{fact_style}_q{q_style}",
            "prompt": f"{facts_text} {question}", "answer": answer,
        }

    # n == 2
    tie = rng.random() < 0.15
    names = rng.sample(pool, 2)
    if tie:
        v = rng.randint(attr["min"], attr["max"])
        values = [v, v]
        form = "relation"
    else:
        values = rng.sample(range(attr["min"], attr["max"] + 1), 2)
        form = rng.choice(["which", "relation"])

    fact_a = render_fact(lang_data, names[0], attr, values[0])
    fact_b = render_fact(lang_data, names[1], attr, values[1])
    join_key = "fact_join_and" if fact_style == 0 else "fact_join_contrast"
    facts_text = lang_data["templates"][join_key].format(fact_a=fact_a, fact_b=fact_b)

    if form == "which":
        q_key = "question_which_2_a" if q_style == 0 else "question_which_2_b"
        question = lang_data["templates"][q_key].format(
            a=names[0], b=names[1], whose=attr["whose"], attr_noun=attr["noun"])
        answer = names[0] if values[0] > values[1] else names[1]
    else:
        q_key = "question_relation_2_a" if q_style == 0 else "question_relation_2_b"
        question = lang_data["templates"][q_key].format(
            a=names[0], b=names[1], attr_noun=attr["noun"], gen=attr["gen"])
        if values[0] > values[1]:
            answer = lang_data["relation_words"]["more"]
        elif values[0] < values[1]:
            answer = lang_data["relation_words"]["less"]
        else:
            answer = lang_data["relation_words"]["equal"]

    return {
        "category": "direct_numeric", "n_entities": 2, "form": form, "attribute": attr_key,
        "entities": names, "values": values,
        "template_id": f"direct_numeric_2_{form}_f{fact_style}_q{q_style}",
        "prompt": f"{facts_text} {question}", "answer": answer,
    }


def gen_chain(lang_data: dict, split: str, entity_pools: dict, rng: random.Random, n_entities: int) -> dict:
    category = "chain_transitive" if n_entities == 3 else "chain_multihop"
    attr_key = rng.choice(list(lang_data["attributes"].keys()))
    attr = lang_data["attributes"][attr_key]
    pool = entity_pools[attr["domain"]][split]
    fact_style, q_style = rng.choice(SPLIT_STYLE_COMBOS[split])

    names = rng.sample(pool, n_entities)
    values = rng.sample(range(attr["min"], attr["max"] + 1), n_entities)

    link_sentences = []
    for i in range(n_entities - 1):
        word = lang_data["relation_words"]["more"] if values[i] > values[i + 1] else lang_data["relation_words"]["less"]
        sent = lang_data["templates"]["link_fact"].format(
            x=names[i], y=names[i + 1], gen=attr["gen"], attr_noun=attr["noun"], word=word)
        if i > 0 and fact_style == 1:
            sent = lang_data["templates"]["chain_join_prefix"] + sent
        link_sentences.append(sent)
    facts_text = " ".join(link_sentences)

    form = rng.choice(["which", "relation"])
    if form == "which":
        extreme = rng.choice(["max", "min"])
        extreme_word = lang_data["extreme_words"][extreme]
        group_word = lang_data["group_words"]["three" if n_entities == 3 else "four"]
        q_key = "question_which_3_a" if q_style == 0 else "question_which_3_b"
        question = lang_data["templates"][q_key].format(
            whose=attr["whose"], attr_noun=attr["noun"], extreme=extreme_word, group_word=group_word)
        target = max(values) if extreme == "max" else min(values)
        answer = names[values.index(target)]
    else:
        if n_entities == 3:
            i, j = 0, 2  # the only non-adjacent pair - forces genuine transitive inference
        else:
            i, j = sorted(rng.sample(range(n_entities), 2))
        x, y = names[i], names[j]
        q_key = "question_relation_chain_a" if q_style == 0 else "question_relation_chain_b"
        question = lang_data["templates"][q_key].format(x=x, y=y, gen=attr["gen"], attr_noun=attr["noun"])
        if values[i] > values[j]:
            answer = lang_data["relation_words"]["more"]
        elif values[i] < values[j]:
            answer = lang_data["relation_words"]["less"]
        else:
            answer = lang_data["relation_words"]["equal"]

    return {
        "category": category, "n_entities": n_entities, "form": form, "attribute": attr_key,
        "entities": names, "values": values,
        "template_id": f"{category}_{form}_f{fact_style}_q{q_style}",
        "prompt": f"{facts_text} {question}", "answer": answer,
    }


def gen_example(lang_data: dict, split: str, entity_pools: dict, rng: random.Random) -> dict:
    category = rng.choice(["direct_numeric", "chain_transitive", "chain_multihop"])
    if category == "direct_numeric":
        return gen_direct_numeric(lang_data, split, entity_pools, rng)
    n_entities = 3 if category == "chain_transitive" else 4
    return gen_chain(lang_data, split, entity_pools, rng, n_entities)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate_split(lang: str, lang_data: dict, split: str, entity_pools: dict,
                    n: int, rng: random.Random) -> list[dict]:
    records = []
    for i in range(n):
        rec = gen_example(lang_data, split, entity_pools, rng)
        rec["id"] = f"{lang}-{split}-{i:06d}"
        rec["lang"] = lang
        records.append(rec)
    return records


def compute_stats(lang: str, entity_pools: dict, records_by_split: dict[str, list[dict]]) -> dict:
    stats = {
        "lang": lang, "seed": SEED,
        "entity_pools": {
            domain: {s: len(pools[s]) for s in ("train", "val", "test")}
            for domain, pools in entity_pools.items()
        },
        "splits": {},
        "template_variety": {},
        "examples": {},
    }
    all_template_ids: dict[str, set[str]] = {}
    for split, records in records_by_split.items():
        by_category: dict[str, int] = {}
        by_form: dict[str, int] = {}
        by_attribute: dict[str, int] = {}
        for r in records:
            by_category[r["category"]] = by_category.get(r["category"], 0) + 1
            by_form[r["form"]] = by_form.get(r["form"], 0) + 1
            by_attribute[r["attribute"]] = by_attribute.get(r["attribute"], 0) + 1
            all_template_ids.setdefault(split, set()).add(r["template_id"])
        stats["splits"][split] = {
            "count": len(records), "by_category": by_category,
            "by_form": by_form, "by_attribute": by_attribute,
        }
        stats["examples"][split] = records[:3]

    template_groups: dict[str, dict[str, set]] = {}
    for split, ids in all_template_ids.items():
        for tid in ids:
            group = tid.rsplit("_f", 1)[0]
            template_groups.setdefault(group, {"train": set(), "val": set(), "test": set()})
            template_groups[group][split].add(tid)
    stats["template_variety"] = {
        group: {s: sorted(ids[s]) for s in ("train", "val", "test")}
        for group, ids in template_groups.items()
    }
    stats["leakage_controls"] = (
        "Entity names/nouns split disjoint 70/15/15 across train/val/test (see "
        "entity_pools above) - a name/object seen at test time never appears in "
        "train. Independently, every (category,form) combination's 4 phrasing "
        "variants (2x2 fact/link-join-style x question-style grid) are split "
        "disjoint the same way (see template_variety above) - val/test always use "
        "wording never seen in train."
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["hindi", "nepali"], default=None,
                     help="Default: generate both languages")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="Default: <lang>/reasoning/data")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    langs = [args.lang] if args.lang else list(LANGS.keys())
    for lang in langs:
        lang_data = LANGS[lang]
        rng = random.Random(args.seed)
        entity_pools = build_entity_pools(lang_data, rng)

        records_by_split = {}
        for split, target in SPLIT_TARGET_SIZE.items():
            records_by_split[split] = generate_split(lang, lang_data, split, entity_pools, target, rng)

        out_dir = args.out_dir or (REPO_ROOT / lang / "reasoning" / "data")
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, records in records_by_split.items():
            path = out_dir / f"{split}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{lang}] wrote {len(records)} examples to {path}", file=sys.stderr)

        stats = compute_stats(lang, entity_pools, records_by_split)
        stats_path = out_dir / "stats.json"
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{lang}] wrote {stats_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
