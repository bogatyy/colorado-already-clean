# Colorado and Bolivia share a neuron, in model after model

Qwen3-32B has a neuron that fires on text about Colorado. The essay [Neural Networks Are Already Clean](https://already-clean.goodfire.com/) (Nick Cammarata, Goodfire) maps everything else that neuron fires on, and one branch of its map is "Bolivia and the Andes".

Is that pairing an accident of one training run? This repository finds each model's own Colorado neuron by a fixed rule, using US text only, and then reads how hard it fires on articles about each of 179 countries.

In Llama 3.3 70B, Gemma 3 27B, Gemma 4 31B and Mistral Small 3.2, as in Qwen3-32B, Bolivia is the country that fires the Colorado neuron most. It is not so in every model: the table below lists all thirteen that could be measured, and the six that could not.

Pictures: **[Colorado Neuron Maps](https://bogatyy.github.io/colorado-already-clean/)** shows 24 of these neurons, one dot per article. Select a map to open it. The page is served from `docs/` in this repository.

## Results

"The Colorado neuron" of a model is the MLP neuron that best separates Colorado articles from other US and general articles (the selection rule is under Method). Share is the neuron's mean firing on a country's articles divided by its mean firing on Colorado articles, so 1.00 means "as hard as on Colorado itself".

| Model | Its Colorado neuron | Bolivia among 179 countries | Bolivia's share (95% range) | 1st in | Top country | Bolivia among 216 places abroad | Bolivia 1st on | Bolivia on the mean of the best 8 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-32B | `L55/N11703P` | **1st** | 0.72 (0.53 to 0.94) | 88% | Bolivia 0.72 | 1st | 6 of 16 | 1st |
| Gemma 3 27B | `L35/N1107P` | **1st** | 0.69 (0.51 to 0.94) | 100% | Bolivia 0.69 | 1st | 9 of 16 | 1st |
| Llama 3.3 70B | `L42/N28598M` | **1st** | 0.46 (0.30 to 0.69) | 100% | Bolivia 0.46 | 1st | 7 of 16 | 1st |
| Mistral Small 3.2 | `L27/N14184M` | **1st** | 0.35 (0.25 to 0.47) | 100% | Bolivia 0.35 | 1st | 3 of 16 | 2nd |
| Gemma 4 31B | `L42/N8677M` | **1st** | 0.11 (0.07 to 0.15) | 86% | Bolivia 0.11 | 3rd (Sichuan 0.29) | 3 of 16 | 1st |
| OLMo-2 7B | `L22/N6538P` | **1st** | 0.10 (0.06 to 0.15) | 75% | Bolivia 0.10 | 1st | 2 of 14 | 1st |
| OLMo-2 13B | `L29/N1368P` | 2nd | 0.93 (0.64 to 1.31) | 6% | Nepal 1.16 | 2nd (Nepal 1.16) | 3 of 16 | 2nd |
| Granite 4.2 30B | `L44/N12921P` | 2nd | 0.05 (0.03 to 0.08) | 9% | Colombia 0.08 | 7th (Sichuan 0.27) | 3 of 16 | 1st |
| Muse Glimmer 30B | `L14/N10142P` | 4th | 0.03 (0.02 to 0.06) | 0% | Nepal 0.17 | 11th (Sichuan 0.83) | 0 of 16 | 7th |
| Marin 32B | `L44/N22385M` | 4th | 0.02 (0.01 to 0.04) | 0% | Austria 0.10 | 9th (Alberta 0.41) | 1 of 16 | 5th |
| Phi-4 | `L26/N17652M` | 7th | 0.23 (0.16 to 0.32) | 0% | Pakistan 1.73 | 13th (Pakistan 1.73) | 2 of 16 | 2nd |
| Qwen3.6-27B | `L55/N16769M` | 12th | 0.01 (0.00 to 0.01) | 0% | Ecuador 0.11 | 17th (Ecuador 0.11) | 0 of 16 | 19th |
| Marin 8B | `L24/N10683M` | 51st | 0.00 (0.00 to 0.00) | 0% | Nepal 0.01 | 75th (Alberta 0.03) | 1 of 16 | 6th |

How to read the columns:

- **Share (95% range)** and **1st in**: the articles are redrawn 2,000 times; the range covers 95% of the redraws, and "1st in" is how often Bolivia stays the top country.
- **Among 216 places abroad** adds 37 regions (Tibet, Sichuan, Alberta, Tyrol and so on) to the 179 countries. Where a region beats Bolivia it is named.
- **Bolivia 1st on** counts, among the model's best Colorado neurons (up to 16), those where Bolivia is the top country with a share of at least 0.1.
- **On the mean of the best 8** is Bolivia's rank among countries when the shares of the model's eight best Colorado neurons are averaged. It shows how far the result depends on the single top neuron.

Across the 13 models, Bolivia is the top country on the Colorado neuron in 6 (Qwen3-32B, Gemma 3 27B, Llama 3.3 70B, Mistral Small 3.2, Gemma 4 31B, OLMo-2 7B). Over all 206 saved Colorado neurons, Bolivia is first on 40 (in 11 models), Nepal is first on 17 (in 8 models), Colombia is first on 11 (in 7 models); no other country is first on more than 5.

Six more models were tried and have no neuron that passes the selection rule, so nothing can be read from them: Pythia 410M, 1.4B, 2.8B, 6.9B and 12B, and Gensyn open-1b.

### What the result is not

- **Not every model.** Bolivia is not the top country on the Colorado neuron of 7 models: OLMo-2 13B (2nd; top country Nepal), Granite 4.2 30B (2nd; top country Colombia), Muse Glimmer 30B (4th; top country Nepal), Marin 32B (4th; top country Austria), Phi-4 (7th; top country Pakistan), Qwen3.6-27B (12th; top country Ecuador), Marin 8B (51st; top country Nepal). 5 of them (OLMo-2 13B, Granite 4.2 30B, Marin 32B, Phi-4, Marin 8B) still have Bolivia first on one or more of their other Colorado neurons (column "Bolivia 1st on").
- **Not unique to Colorado.** The probe selects neurons for four other states in the same way. Their neurons keep foreign company too:

  | State | Neurons (models) | Country most often 1st | Next |
  |---|---|---|---|
  | Colorado | 206 (13) | Bolivia: 40 neurons, 11 models | Nepal: 17 neurons, 8 models; Colombia: 11 neurons, 7 models |
  | Utah | 205 (13) | Tonga: 8 neurons, 6 models | Bolivia: 8 neurons, 1 model; Samoa: 7 neurons, 6 models |
  | New Mexico | 201 (13) | Venezuela: 13 neurons, 7 models | Uruguay: 11 neurons, 8 models; Bolivia: 10 neurons, 8 models |
  | Wyoming | 185 (13) | Mongolia: 13 neurons, 6 models | Bhutan: 7 neurons, 4 models; Kyrgyzstan: 7 neurons, 4 models |
  | Montana | 200 (14) | Montenegro: 32 neurons, 13 models | Mongolia: 22 neurons, 12 models; North Macedonia: 10 neurons, 6 models |

  (Montana counts 14 models because Pythia 12B, which has no Colorado neuron that passes the rule, has Montana ones.)

  So "a US state's neuron also fires on some country" is the rule. What is specific to Colorado is which country, and that the same one comes up across unrelated models. Montana's pairing with Montenegro and Mongolia is at least as consistent, but those names share their first letters with Montana; Colorado and Bolivia share nothing in spelling. The other four states have fewer selection articles than Colorado (4 to 11, against 33), so their neurons are chosen less reliably.
- **Not explained.** Nothing here says why Bolivia. The essay's map describes the Bolivia branch as "a mountain-state analogue of Colorado". Both places are mountainous, but so are many countries that fire these neurons weakly.
- **Not large in every model.** In Gemma 4 31B and OLMo-2 7B Bolivia is first with a share near 0.1: the neuron fires on Bolivia articles at about a tenth of its Colorado level, and on every other country less. In Gemma 4 two Chinese provinces, Sichuan and Yunnan, fire it harder than Bolivia does.
- **Not always one clear neuron.** In Mistral Small 3.2 the five best Colorado neurons are within 0.02 of each other in selection score. The top one and the fourth put Bolivia first (shares 0.35 and 0.58); the second and third put it third.

## Method

**A neuron** is one hidden unit of an MLP block. Its activation is the gated product that feeds the block's output layer. It is counted only at tokens where the gate is open, and the two signs are treated as two neurons (positive pole `P`, negative pole `M`), so `L55/N11703P` is layer 55, unit 11703, positive pole.

**An article's score** on a neuron is the mean of the neuron's eight strongest token activations in that article.

**Selection** uses 316 hand-listed Wikipedia articles: 236 about US states, cities and landmarks (33 of them about Colorado) and 80 general ones (Photosynthesis, Jazz, Calculus). Every MLP neuron in the model is scored by the correlation between its article scores and a 0/1 label "this article is about Colorado". A neuron is kept if the correlation is at least 0.5 and its highest score on any general article is at most 0.3 of its mean score on Colorado articles. The kept neuron with the highest correlation is the Colorado neuron; the best 16 are saved. No foreign article takes part.

**Reading** uses a separate, random sample: 7,239 Wikipedia articles drawn with a fixed seed from the March 2022 English dump (2,000 at random, plus 24 random articles for each of 179 countries, 49 US states and 37 other regions), the first 900 characters of each. No article in it was chosen by hand. An article belongs to the place it names first. Articles that contain "Colorad" or "Denver" are left out of every country's side, so a Bolivian article that mentions Colorado cannot produce the result.

Selection and reading are one script, [`probe/tiecheck.py`](probe/tiecheck.py), with no per-model settings.

## Reproduce

### 1. Recount from the saved readings (no GPU, a few minutes)

`results/tie/` holds, for every model, the selected neurons (`.json`) and their score on each of the 7,239 articles (`.npz`).

```bash
pip install numpy
python analysis/summary.py          # one line per model: the Colorado neuron and Bolivia's rank
python analysis/claims.py           # every neuron, with resampling intervals; writes results/claims.json
python analysis/states.py           # the other-state control; writes results/claims_states.json
```

The output of these three commands, as run for this README, is in `results/summary.txt`, `results/claims.txt` and `results/states.txt`.

### 2. Measure a model yourself (one GPU)

The saved results came from 96 GB GPUs (the 24B to 32B models used 50 to 70 GB) and, for Llama 3.3 70B, a 180 GB GPU (about 140 GB used), with the weights in bfloat16. A run takes 5 to 12 minutes once the weights are downloaded. Re-measuring six models for this release (Qwen3-32B, Qwen3.6-27B, Gemma 3, Gemma 4, Mistral Small, Llama 3.3 70B) cost about $3.60 of rented GPU time in total.

```bash
pip install -U "transformers>=5.5" accelerate "huggingface_hub[hf_xet]" safetensors sentencepiece protobuf mistral-common
cd probe
python tiecheck.py --model Qwen/Qwen3-32B --hand ../results/corpus.jsonl --docs ../results/blind_corpus.jsonl --out ../results/tie/qwen3 --nodir
```

`--nodir` skips an extra measurement that is not used here. [`probe/run_tie_some.sh`](probe/run_tie_some.sh) lists the model id behind every tag and is the script the saved results came from (it assumes a `/workspace` volume). Exact model revisions and library versions are in [`results/tie/revisions.json`](results/tie/revisions.json).

Runs repeat, in five of six cases exactly. Qwen3-32B, Llama 3.3 70B and Gemma 3 27B were each measured twice, on separately rented machines: the same neurons were selected and every score was bit-identical. Gemma 4 31B and Mistral Small 3.2 were first measured with an earlier script; the scores of the neurons both scripts read are bit-identical too. Qwen3.6-27B is the exception: its second run selected the same neurons, but its scores differed slightly (correlation between the two runs at least 0.9997 for every neuron). Its row in the table is the same either way. The cause was not tracked down.

### 3. Rebuild the circle maps

```bash
pip install numpy "transformers>=5.5" sentencepiece protobuf mistral-common
python analysis/build_page.py       # rewrites docs/data from results/tie
```

The page is static and GitHub Pages serves it from `docs/`. To see it locally, open `docs/index.html` through any web server (for example `python -m http.server` in `docs/`). Its percentages are the shares defined above, times 100, ranked over all 216 places abroad (the 179 countries and the 37 regions).

## Files

| Path | What it is |
|---|---|
| `probe/tiecheck.py` | The measurement: selection, reading, and a "which state is most similar to X" question that is saved but not used here |
| `probe/common.py`, `probe/places.py` | Tokenising helpers; the lists of states, countries and hand-listed articles |
| `probe/build_corpus.py`, `probe/make_blind_corpus.py` | How the two text sets were built from the Wikipedia dump |
| `results/corpus.jsonl` | The hand-listed articles (756; the probe uses the 316 US and general ones) |
| `results/blind_corpus.jsonl` | The 7,239-article sample |
| `results/tie/<tag>.json`, `.npz` | Per model: selected neurons, and their scores on every sampled article |
| `analysis/models.py` | Model names and tags |
| `analysis/summary.py`, `claims.py`, `states.py` | The recounts above |
| `results/claims.json`, `claims_states.json`, `results/*.txt` | Saved output of the recounts |
| `analysis/build_page.py`, `docs/` | The circle maps: the script that builds their data, and the page itself |

## Caveats

- Llama 3.3 70B and Gemma 3 27B were loaded from ungated mirrors (`unsloth/…`) of the gated originals. That the mirrors are byte-identical to the originals was not verified.
- Each country has about 24 articles (21 to 41), all from one random draw. The intervals in the table come from redrawing those same articles; a fresh sample from Wikipedia could differ more.
- Only the first 900 characters of each article are read, at most 224 tokens.
- The Colorado neuron is defined by one rule. The count over each model's 16 best neurons is given so the result does not rest on the single top neuron.
- Instruction-tuned models were read on plain text, without a chat template.

## Credits and licences

The question comes from [Neural Networks Are Already Clean](https://already-clean.goodfire.com/) by Nick Cammarata, with contributions from Curt Tigges and Tarun Gogineni (Goodfire). The circle maps follow the essay's way of drawing a neuron.

The article text in `results/corpus.jsonl` and `results/blind_corpus.jsonl` comes from English Wikipedia (the 2022-03-01 dump via the Hugging Face `wikipedia` dataset; eight hand-listed articles from the live site) and is available under the [Creative Commons Attribution-ShareAlike licence](https://creativecommons.org/licenses/by-sa/4.0/). Each record carries its article title.
