import re
from typing import List, Dict, Optional

class CropHealthAdvisoryEngine:
    

    def __init__(self):
      
        self.knowledge_base = {

           
            "bipolaris": { 
                "type": "Fungal disease",
                "immediate": [
                    "Identify stressed field zones; brown spot severity increases under nutrient stress.",
                    "Maintain uniform soil moisture to reduce lesion expansion."
                ],
                "fertilizer": [
                    "Correct Potassium and Zinc deficiencies immediately.",
                    "Apply Silicon sources to strengthen leaf epidermis."
                ],
                "control": [
                    "Apply Propiconazole or Tebuconazole as foliar spray.",
                    "Rotate with Mancozeb in the next spray cycle."
                ],
                "app_guide": [
                    "Ensure full leaf surface coverage.",
                    "Spray during low wind and moderate humidity."
                ],
                "preventive": [
                    "Use balanced fertilization in early crop stages.",
                    "Avoid drought stress during tillering."
                ]
            },

            "ustilaginoidea": {  # False smut
                "type": "Fungal disease",
                "immediate": [
                    "Target panicle initiation stage infections early.",
                    "Remove heavily infected panicles where feasible."
                ],
                "fertilizer": [
                    "Avoid late-season urea application.",
                    "Increase Potassium to strengthen panicle tissues."
                ],
                "control": [
                    "Apply Triazole fungicides during booting stage.",
                    "Rotate with QoI fungicides if reapplication needed."
                ],
                "app_guide": [
                    "Foliar spray targeting emerging panicles.",
                    "Reapply after 10–12 days if humid conditions persist."
                ],
                "preventive": [
                    "Adopt tolerant varieties in next season.",
                    "Avoid excess Nitrogen before flowering."
                ]
            },

            # ---------- SOLANACEAE ----------
            "phytophthora": {  # Late blight
                "type": "Oomycete disease",
                "immediate": [
                    "Remove infected foliage to limit sporangia spread.",
                    "Monitor cool and humid weather closely."
                ],
                "fertilizer": [
                    "Reduce Nitrogen-driven canopy density.",
                    "Apply Phosphite-based immunity boosters."
                ],
                "control": [
                    "Use Metalaxyl-M + Mancozeb combinations.",
                    "Alternate with Cymoxanil-based fungicides."
                ],
                "app_guide": [
                    "Spray before rainfall events.",
                    "Ensure coverage of lower canopy leaves."
                ],
                "preventive": [
                    "Use certified disease-free seed material.",
                    "Destroy crop residues post-harvest."
                ]
            },

            # ---------- WHEAT ----------
            "puccinia": {  # Rusts
                "type": "Fungal disease",
                "immediate": [
                    "Scout fields for active pustule development.",
                    "Spot-treat infection hotspots immediately."
                ],
                "fertilizer": [
                    "Avoid late Nitrogen top-dressing.",
                    "Maintain balanced NPK nutrition."
                ],
                "control": [
                    "Apply Tebuconazole or Propiconazole.",
                    "Ensure timely first spray to halt spread."
                ],
                "app_guide": [
                    "Spray during calm weather for uniform deposition."
                ],
                "preventive": [
                    "Use resistant cultivars.",
                    "Eliminate alternate host plants nearby."
                ]
            },

            # ---------- GENERIC VIRAL ----------
            "virus": {
                "type": "Viral disease",
                "immediate": [
                    "Rogue infected plants promptly to prevent spread.",
                    "Control vector insects immediately."
                ],
                "fertilizer": [
                    "Support healthy plants with micronutrients.",
                    "Avoid excessive Nitrogen."
                ],
                "control": [
                    "Manage vectors using selective systemic insecticides.",
                    "Deploy yellow sticky traps for monitoring."
                ],
                "app_guide": [
                    "Target sprays to leaf undersides where vectors reside."
                ],
                "preventive": [
                    "Use virus-free planting material.",
                    "Maintain weed-free field borders."
                ]
            },

            # ---------- BACTERIAL ----------
            "xanthomonas": {
                "type": "Bacterial disease",
                "immediate": [
                    "Avoid field operations when foliage is wet.",
                    "Remove and destroy infected plant parts."
                ],
                "fertilizer": [
                    "Avoid excess Nitrogen fertilization.",
                    "Ensure adequate Potassium supply."
                ],
                "control": [
                    "Apply Copper-based bactericides where permitted.",
                    "Use biological antagonists as preventive tools."
                ],
                "app_guide": [
                    "Apply preventively before heavy rains."
                ],
                "preventive": [
                    "Use certified clean seeds.",
                    "Disinfect tools and equipment."
                ]
            }
        }

        # =====================================================
        # 2. HEURISTIC FALLBACKS (FOR 200+ DISEASES)
        # =====================================================
        self.heuristics = [
            (r"virus|mosaic|curl|wilt", "Viral disease", self._viral_fallback),
            (r"xanthomonas|pseudomonas|ralstonia|erwinia", "Bacterial disease", self._bacterial_fallback),
            (r"borer|aphid|bug|beetle|thrips|mite|weevil|moth|larva", "Insect pest", self._insect_fallback),
            (r"nematode|meloidogyne", "Nematode infestation", self._nematode_fallback),
            (r".*", "Fungal disease", self._fungal_fallback)
        ]

    # =====================================================
    # FALLBACK GENERATORS (SAFE, AGRONOMIC)
    # =====================================================
    def _fungal_fallback(self):
        return {
            "type": "Fungal disease",
            "immediate": ["Reduce canopy humidity.", "Remove severely infected tissues."],
            "fertilizer": ["Avoid excess Nitrogen.", "Ensure Potassium sufficiency."],
            "control": ["Apply broad-spectrum fungicides.", "Rotate modes of action."],
            "app_guide": ["Ensure uniform foliar coverage."],
            "preventive": ["Practice crop rotation.", "Use tolerant varieties."]
        }

    def _bacterial_fallback(self):
        return {
            "type": "Bacterial disease",
            "immediate": ["Limit field activity during wet conditions."],
            "fertilizer": ["Avoid lush growth via excess Nitrogen."],
            "control": ["Apply copper-based bactericides preventively."],
            "app_guide": ["Spray during dry intervals."],
            "preventive": ["Use clean planting material."]
        }

    def _viral_fallback(self):
        return {
            "type": "Viral disease",
            "immediate": ["Remove infected plants.", "Control vectors urgently."],
            "fertilizer": ["Support plant vigor with micronutrients."],
            "control": ["Target insect vectors only."],
            "app_guide": ["Focus on vector habitats."],
            "preventive": ["Adopt resistant varieties."]
        }

    def _insect_fallback(self):
        return {
            "type": "Insect pest",
            "immediate": ["Assess pest life stage before intervention."],
            "fertilizer": ["Avoid Nitrogen excess."],
            "control": ["Apply crop-approved insecticides.", "Rotate insecticide classes."],
            "app_guide": ["Spray during peak pest activity."],
            "preventive": ["Encourage natural enemies."]
        }

    def _nematode_fallback(self):
        return {
            "type": "Nematode infestation",
            "immediate": ["Restrict soil movement."],
            "fertilizer": ["Increase organic matter."],
            "control": ["Apply nematicides if permitted."],
            "app_guide": ["Soil-directed application required."],
            "preventive": ["Rotate with non-host crops."]
        }

    # =====================================================
    # CORE RESOLUTION LOGIC
    # =====================================================
    def _resolve(self, scientific_name: str) -> Dict:
        s = scientific_name.lower()

        for key, data in self.knowledge_base.items():
            if key in s:
                return data

        for pattern, _, generator in self.heuristics:
            if re.search(pattern, s):
                return generator()

        return self._fungal_fallback()

    # =====================================================
    # PUBLIC API (DO NOT CHANGE NAMES)
    # =====================================================
    def infer_disease_type(self, scientific_name: str) -> str:
        return self._resolve(scientific_name).get("type", "Plant disease")

    def generate_immediate_actions(self, scientific_name: str) -> List[str]:
        return self._resolve(scientific_name).get("immediate", [])

    def generate_fertilizer_plan(self, scientific_name: str) -> List[str]:
        return self._resolve(scientific_name).get("fertilizer", [])

    def generate_control_plan(self, scientific_name: str) -> List[str]:
        return self._resolve(scientific_name).get("control", [])

    def generate_application_guidelines(self, scientific_name: str) -> List[str]:
        return self._resolve(scientific_name).get("app_guide", [])

    def generate_preventive_care(self, scientific_name: str) -> List[str]:
        return self._resolve(scientific_name).get("preventive", [])



_engine = CropHealthAdvisoryEngine()

def infer_disease_type(scientific_name: str) -> str:
    return _engine.infer_disease_type(scientific_name)

def generate_immediate_actions(scientific_name: str) -> List[str]:
    return _engine.generate_immediate_actions(scientific_name)

def generate_fertilizer_plan(scientific_name: str) -> List[str]:
    return _engine.generate_fertilizer_plan(scientific_name)

def generate_control_plan(scientific_name: str) -> List[str]:
    return _engine.generate_control_plan(scientific_name)

def generate_application_guidelines(scientific_name: str) -> List[str]:
    return _engine.generate_application_guidelines(scientific_name)

def generate_preventive_care(scientific_name: str) -> List[str]:
    return _engine.generate_preventive_care(scientific_name)
