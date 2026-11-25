"""The models of the study: display name -> tag of its result files (results/tie/<tag>.json and .npz).

Hugging Face ids and the exact revisions are in results/tie/revisions.json; probe/run_tie_some.sh maps tags to ids.
The first thirteen have Colorado neurons that pass the selection rule. The last six (Pythia, Gensyn) have none,
so nothing can be measured in them; they are listed so the count of models tried is complete.
"""
MODELS = {
    "Qwen3-32B": "qwen3",
    "Llama 3.3 70B": "llama33_70b",
    "Gemma 3 27B": "gemma3_27b",
    "Gemma 4 31B": "gemma4",
    "Mistral Small 3.2": "mistral32",
    "OLMo-2 13B": "olmo13b",
    "OLMo-2 7B": "olmo7b",
    "Granite 4.2 30B": "granite42_30b",
    "Phi-4": "phi4",
    "Muse Glimmer 30B": "muse30b",
    "Marin 32B": "marin32b",
    "Marin 8B": "marin8b",
    "Qwen3.6-27B": "qwen36",
    "Pythia 12B": "pythia12b",
    "Pythia 6.9B": "pythia6_9b",
    "Pythia 2.8B": "pythia2_8b",
    "Pythia 1.4B": "pythia1_4b",
    "Pythia 410M": "pythia410m",
    "Gensyn open-1b": "gensyn",
}
