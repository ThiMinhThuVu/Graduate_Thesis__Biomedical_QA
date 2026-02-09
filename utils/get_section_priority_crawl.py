def get_sections_for_target_type(target_type: str):
    target_type = target_type.lower().strip()
    return TARGET_TYPE_SECTIONS.get(
        target_type,
        TARGET_TYPE_SECTIONS["other"]
    )


TARGET_TYPE_SECTIONS = {
  "condition": [
    "diagnosis",
    "signs and symptoms",
    "causes",
    "epidemiology",
    "treatment",
    # "fossil record",
    "risk factors",
  ],
  "gene": [
    "causes",
    "genetics",
    "diagnosis",
    "diseases and disorders"
  ],
  "disease": [
    "causes",
    "signs and symptoms",
    "diagnosis",
    "epidemiology",
    "treatment"
  ],
  "medication": [
    "treatment",
    "toxicity"
  ],
  "disorder": [
    "clinical significance",
    "cause",
    "diagnosis",
    "signs and symptoms",
    "treatment",
    "epidemiology"
  ],
  "syndrome": [
    "function",
    "causes",
    "signs and symptoms",
    "diagnosis"
  ],
  "chromosome": [
    "genetics structure",
    "genetics",
    "cause",
    "chromosomal disorder",
    "pathophysiology"
  ],
  "protein": [
    "causes",
    "pathophysiology",
    "viral proteins"
  ],
  "person": [
    "biography",
    "history",
    "diagnosis"
  ],
  "symptom": [
    "signs and symptoms",
    "cause",
    "diagnosis"
  ],
  "virus": [
    "causes",
    "related disorders",
    "epidemiology"
  ],
  "gene mutation": [
    "genetics",
    "diagnosis",
    "causes"
  ],
  "drug": [
    "treatment",
    "clinical significance"
  ],
  "enzyme": [
    "genetics",
    "pathology",
    "diagnosis"
  ],
  "tumor": [
    "signs and symptom",
    "diagnosis",
    "causes",
    "treatment",
    "epidemiology"
  ],
  "anatomical structure": [
    "pathophysiology",
    "diagnosis",
    "history"
  ],
  "nerve": [
    "signs and symptoms"
  ],
  "antibiotic": [
    "medications",
    "treatment"
  ],
  "organ": [
    "signs and symptoms"
  ],
  "suborder": [
    "fossil record"
  ],
  "other": [
        "cause",
        "causes",
        "diagnosis",
    ],
}