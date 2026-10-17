"""Stage 1 & 2 guardrails: zero-shot CLIP scene understanding.

CLIP scores the image against natural-language prompts that are grouped into
buckets (raw produce, packaged food, prepared food, non-food). Summing the
softmax probability per bucket gives:

* Stage 1 - is this a food item at all?        (out-of-distribution filter)
* Stage 2 - raw produce or packaged? solid or liquid?   (category validation)
* Produce identity - which fruit/vegetable is it, and is it one the spoilage
  CNN was actually trained on? (stops a mango being "graded" as an apple)
"""
from functools import lru_cache

import torch
from PIL import Image

from . import config
from .models import DEVICE, get_clip

SCENE_PROMPTS = {
    "raw_produce": [
        "a photo of a fresh fruit",
        "a photo of a fresh vegetable",
        "a photo of a rotten fruit",
        "a photo of a rotten or moldy vegetable",
        "a close-up photo of a single fruit or vegetable",
        "a photo of fruits and vegetables at an Indian market",
    ],
    "packaged_solid": [
        "a photo of a packet of chips or namkeen",
        "a photo of a packet of biscuits or cookies",
        "a photo of a chocolate bar wrapper",
        "a photo of a packet of instant noodles",
        "a photo of a nutrition facts table printed on a food packet",
        "a photo of the back of a food package with ingredients text",
        "a photo of a box of breakfast cereal",
        "a photo of a jar of jam, spread or pickle",
    ],
    "packaged_liquid": [
        "a photo of a bottle of fruit juice or soft drink",
        "a photo of a can of soda or energy drink",
        "a photo of a tetra pack carton of juice or milk",
        "a photo of a bottle of flavoured milk",
    ],
    "packaged_exempt": [
        "a photo of a bottle of cooking oil",
        "a photo of a jar of ghee",
        "a photo of a bottle or jar of honey",
        "a photo of a packet of dry fruits like almonds or cashews",
    ],
    "prepared_food": [
        "a photo of a cooked meal on a plate",
        "a photo of a bowl of curry, dal or rice",
        "a photo of a pizza, burger or sandwich",
        "a photo of street food",
    ],
    "non_food": [
        "a photo of a person",
        "a selfie",
        "a photo of a dog, cat or other animal",
        "a photo of a shoe",
        "a photo of clothing",
        "a photo of a shampoo, soap or cosmetic bottle",
        "a photo of a detergent or cleaning product",
        "a photo of medicine tablets or a medicine box",
        "a photo of a mobile phone, laptop or electronics",
        "a photo of furniture or the inside of a room",
        "a photo of a car or motorbike",
        "a photo of a building or a street",
        "a landscape photo",
        "a screenshot of a document or a web page",
        "a photo of a toy",
        "a photo of a plant or flower",
        "a cartoon or drawing",
    ],
}

# The six types the CNN knows, plus look-alikes it must NOT be asked to grade.
PRODUCE_NAMES = {
    "apple": ["apple"],
    "banana": ["banana"],
    "bitter_gourd": ["bitter gourd", "karela"],
    "capsicum": ["capsicum", "bell pepper"],
    "orange": ["orange"],
    "tomato": ["tomato"],
    "mango": ["mango"],
    "grapes": ["bunch of grapes"],
    "potato": ["potato"],
    "onion": ["onion"],
    "carrot": ["carrot"],
    "cucumber": ["cucumber"],
    "cauliflower": ["cauliflower"],
    "cabbage": ["cabbage"],
    "brinjal": ["brinjal", "eggplant"],
    "lemon": ["lemon"],
    "pomegranate": ["pomegranate"],
    "watermelon": ["watermelon"],
    "papaya": ["papaya"],
    "guava": ["guava"],
    "strawberry": ["strawberry"],
    "pineapple": ["pineapple"],
    "okra": ["okra", "lady finger vegetable"],
    "leafy_greens": ["spinach or leafy greens"],
    "green_chilli": ["green chilli"],
    "pear": ["pear"],
    "coconut": ["coconut"],
}


def _features(output) -> torch.Tensor:
    # transformers may return a tensor or a ModelOutput depending on version.
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if value is not None:
            return value
    raise TypeError(f"Unexpected CLIP output type: {type(output)}")


@lru_cache(maxsize=None)
def _text_bank(kind: str):
    """Encode every prompt once. Returns (normalised embeddings, group per prompt)."""
    model, processor = get_clip()
    prompts, groups = [], []
    if kind == "scene":
        for group, texts in SCENE_PROMPTS.items():
            prompts += texts
            groups += [group] * len(texts)
    else:
        for name, aliases in PRODUCE_NAMES.items():
            for alias in aliases:
                prompts += [f"a photo of a {alias}", f"a photo of a rotten {alias}"]
                groups += [name, name]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        emb = _features(model.get_text_features(**inputs))
    return torch.nn.functional.normalize(emb, dim=-1), groups


def _image_embedding(image: Image.Image) -> torch.Tensor:
    model, processor = get_clip()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = _features(model.get_image_features(**inputs))
    return torch.nn.functional.normalize(emb, dim=-1)


def _group_probs(image_emb: torch.Tensor, kind: str):
    model, _ = get_clip()
    text_emb, groups = _text_bank(kind)
    logits = model.logit_scale.exp() * image_emb @ text_emb.T
    probs = logits.softmax(dim=-1)[0].float().cpu().tolist()
    totals: dict[str, float] = {}
    for g, p in zip(groups, probs):
        totals[g] = totals.get(g, 0.0) + p
    best_prompt_idx = max(range(len(probs)), key=probs.__getitem__)
    return totals, best_prompt_idx


def analyse_scene(image: Image.Image) -> dict:
    """Run Stage 1 and the visual half of Stage 2 on one image."""
    image_emb = _image_embedding(image)
    totals, best_idx = _group_probs(image_emb, "scene")
    all_prompts = [p for texts in SCENE_PROMPTS.values() for p in texts]

    raw = totals["raw_produce"]
    packaged = totals["packaged_solid"] + totals["packaged_liquid"] + totals["packaged_exempt"]
    prepared = totals["prepared_food"]
    food = raw + packaged + prepared

    if raw >= packaged and raw >= prepared:
        visual_category = "raw_produce"
    elif packaged >= prepared:
        visual_category = max(
            ("packaged_solid", "packaged_liquid", "packaged_exempt"), key=totals.__getitem__
        )
    else:
        visual_category = "prepared_food"

    result = {
        "food_probability": round(food, 4),
        "group_probabilities": {k: round(v, 4) for k, v in totals.items()},
        "raw_probability": round(raw, 4),
        "packaged_probability": round(packaged, 4),
        "visual_category": visual_category,
        "best_description": all_prompts[best_idx],
    }
    if visual_category == "raw_produce" or raw > 0.2:
        result["produce_identity"] = identify_produce(image_emb)
    return result


def identify_produce(image_emb: torch.Tensor) -> dict:
    totals, _ = _group_probs(image_emb, "produce")
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_prob = ranked[0]
    supported_prob = sum(totals[name] for name in config.SUPPORTED_PRODUCE)
    return {
        "top": top_name,
        "probability": round(top_prob, 4),
        "supported": top_name in config.SUPPORTED_PRODUCE,
        "supported_probability": round(supported_prob, 4),
        "top5": [{"name": n, "probability": round(p, 4)} for n, p in ranked[:5]],
    }
