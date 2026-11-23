# Applied Machine Learning for Civil and Environmental Engineering

Slides and Python notebooks for a 28-lecture course in applied machine learning, built on the
problems and public data of civil and environmental engineering. Every lecture starts from an
engineering question. Each method is derived, built from scratch, and checked against a reference
library before a library or an AI assistant is trusted with it.

The course design, with its learning goals, schedule, homework, and grading, is at
<https://wtorresc.github.io/data/cee-aml.pdf>.

## What is here

| Path | Contents |
|---|---|
| `lectures/build/` | The 28 finished decks, as PDF and PowerPoint. |
| `lectures/` | Deck sources in Markdown, one figure script per lecture, diagrams, and the build tools. Every number on a slide comes from a figure script. |
| `notebooks/` | One notebook per lecture, written in marimo (reactive `.py` files). `00_index.md` maps the concepts across lectures. |
| `ipynb/` | The same notebooks exported to Jupyter. |
| `scripts/` | One `fetch_*.sh` per dataset. Each states its source and licence and checks what it downloads. |
| `pixi.toml`, `pixi.lock` | The pinned environment the numbers were made with. |

Datasets are not stored in this repository. Each fetch script downloads its data from the original
source.

## Lectures

### Unit 1. Learning from engineering data

| # | Lecture | Engineering question | Files |
|---:|---|---|---|
| 1 | When Predictions Carry Loads | What would you need before signing a model's output? | [slides](lectures/build/cee_l01_framing.pdf), [pptx](lectures/build/cee_l01_framing.pptx), [notebook](ipynb/cee_l01_framing.ipynb) |
| 2 | Nearest Neighbors and Representation | Will this concrete mix reach its specified strength? | [slides](lectures/build/cee_l02_nearest_neighbors.pdf), [pptx](lectures/build/cee_l02_nearest_neighbors.pptx), [notebook](ipynb/cee_l02_nearest_neighbors.ipynb) |
| 3 | Generalization and Validation | How wrong will the model be on the next bridge? | [slides](lectures/build/cee_l03_generalization.pdf), [pptx](lectures/build/cee_l03_generalization.pptx), [notebook](ipynb/cee_l03_generalization.ipynb) |
| 4 | Retrieval, Clustering, and Modes | Which buildings should get an after-hours visit? | [slides](lectures/build/cee_l04_load_profiles.pdf), [pptx](lectures/build/cee_l04_load_profiles.pptx), [notebook](ipynb/cee_l04_load_profiles.ipynb) |
| 5 | Least Squares and Regularization | How much cement does a 35 MPa mix need? | [slides](lectures/build/cee_l05_least_squares.pdf), [pptx](lectures/build/cee_l05_least_squares.pptx), [notebook](ipynb/cee_l05_least_squares.ipynb) |
| 6 | Linear Classifiers and Costs | Which bridges will be poor in four years? | [slides](lectures/build/cee_l06_linear_classifiers.pdf), [pptx](lectures/build/cee_l06_linear_classifiers.pptx), [notebook](ipynb/cee_l06_linear_classifiers.ipynb) |
| 7 | Mixtures, EM, and Naive Bayes | When does the freeway break down? | [slides](lectures/build/cee_l07_mixtures_bayes.pdf), [pptx](lectures/build/cee_l07_mixtures_bayes.pptx), [notebook](ipynb/cee_l07_mixtures_bayes.ipynb) |
| 8 | Trees, Forests, and Their Explanations | What makes a pavement wear out fast? | [slides](lectures/build/cee_l08_trees_explanations.pdf), [pptx](lectures/build/cee_l08_trees_explanations.pptx), [notebook](ipynb/cee_l08_trees_explanations.ipynb) |

### Unit 2. Uncertainty, risk, and decisions

| # | Lecture | Engineering question | Files |
|---:|---|---|---|
| 9 | Outliers and Robust Estimation | When did the pipe burst? | [slides](lectures/build/cee_l09_robust_estimation.pdf), [pptx](lectures/build/cee_l09_robust_estimation.pptx), [notebook](ipynb/cee_l09_robust_estimation.ipynb) |
| 10 | Calibration and Quantile Regression | What strength can we count on? | [slides](lectures/build/cee_l10_calibration_quantiles.pdf), [pptx](lectures/build/cee_l10_calibration_quantiles.pptx), [notebook](ipynb/cee_l10_calibration_quantiles.ipynb) |
| 11 | Conformal Prediction and Shift | How sure is next week's river forecast? | [slides](lectures/build/cee_l11_conformal_shift.pdf), [pptx](lectures/build/cee_l11_conformal_shift.pptx), [notebook](ipynb/cee_l11_conformal_shift.ipynb) |
| 12 | Gaussian Processes for Optimization | Which valve settings leak least? | [slides](lectures/build/cee_l12_gaussian_processes.pdf), [pptx](lectures/build/cee_l12_gaussian_processes.pptx), [notebook](ipynb/cee_l12_gaussian_processes.ipynb) |

### Unit 3. Deep learning for sensing and inspection

| # | Lecture | Engineering question | Files |
|---:|---|---|---|
| 13 | Stochastic Gradient Descent | How many decks will lose a point? | [slides](lectures/build/cee_l13_stochastic_gradient.pdf), [pptx](lectures/build/cee_l13_stochastic_gradient.pptx), [notebook](ipynb/cee_l13_stochastic_gradient.ipynb) |
| 14 | Neural Networks and Backpropagation | Can a network stand in for the simulator? | [slides](lectures/build/cee_l14_neural_networks.pdf), [pptx](lectures/build/cee_l14_neural_networks.pptx), [notebook](ipynb/cee_l14_neural_networks.ipynb) |
| 15 | Convolutions and Transfer Learning | Can a network flag the cracked tiles? | [slides](lectures/build/cee_l15_convolutional_networks.pdf), [pptx](lectures/build/cee_l15_convolutional_networks.pptx), [notebook](ipynb/cee_l15_convolutional_networks.ipynb) |
| 16 | Segmentation and Measuring | How wide is the crack, and which roads are under water? | [slides](lectures/build/cee_l16_segmentation.pdf), [pptx](lectures/build/cee_l16_segmentation.pptx), [notebook](ipynb/cee_l16_segmentation.ipynb) |

### Unit 4. Foundation models and time series

| # | Lecture | Engineering question | Files |
|---:|---|---|---|
| 17 | Tokens, Attention, and Transformers | Can a transformer screen the crack tiles better? | [slides](lectures/build/cee_l17_transformers.pdf), [pptx](lectures/build/cee_l17_transformers.pptx), [notebook](ipynb/cee_l17_transformers.ipynb) |
| 18 | Vision Foundation Models | Can the screen start before anyone labels a tile? | [slides](lectures/build/cee_l18_vision_foundation.pdf), [pptx](lectures/build/cee_l18_vision_foundation.pptx), [notebook](ipynb/cee_l18_vision_foundation.ipynb) |
| 19 | Language Models and Retrieval | Can a model answer a crew's question from the manual? | [slides](lectures/build/cee_l19_language_models.pdf), [pptx](lectures/build/cee_l19_language_models.pptx), [notebook](ipynb/cee_l19_language_models.ipynb) |
| 20 | 1-D Signals, CNNs, and LSTMs | Will the river flood tomorrow? Is the bridge still the bridge? | [slides](lectures/build/cee_l20_time_series.pdf), [pptx](lectures/build/cee_l20_time_series.pptx), [notebook](ipynb/cee_l20_time_series.ipynb) |

### Unit 5. Structure and physics

| # | Lecture | Engineering question | Files |
|---:|---|---|---|
| 21 | Graph Neural Networks | Can 33 sensors tell the pressure at 782 junctions? | [slides](lectures/build/cee_l21_graph_networks.pdf), [pptx](lectures/build/cee_l21_graph_networks.pptx), [notebook](ipynb/cee_l21_graph_networks.ipynb) |
| 22 | Point Clouds | Was the lab built as modelled? | [slides](lectures/build/cee_l22_point_clouds.pdf), [pptx](lectures/build/cee_l22_point_clouds.pptx), [notebook](ipynb/cee_l22_point_clouds.ipynb) |
| 23 | Physics-Informed Networks | When can the road be paved? | [slides](lectures/build/cee_l23_physics_informed.pdf), [pptx](lectures/build/cee_l23_physics_informed.pptx), [notebook](ipynb/cee_l23_physics_informed.ipynb) |
| 24 | Learning Dynamics from Data | Will the pond hold? Can a network route the flood? | [slides](lectures/build/cee_l24_learning_dynamics.pdf), [pptx](lectures/build/cee_l24_learning_dynamics.pptx), [notebook](ipynb/cee_l24_learning_dynamics.ipynb) |

### Unit 6. Agents, decisions, and the profession

| # | Lecture | Engineering question | Files |
|---:|---|---|---|
| 25 | Agents, Tools, and Checks | Can a small model answer from the hydraulic model? What must it pass first? | [slides](lectures/build/cee_l25_agents.pdf), [pptx](lectures/build/cee_l25_agents.pptx), [notebook](ipynb/cee_l25_agents.ipynb) |
| 26 | Decisions Over Time | Markov decision processes and reinforcement learning: when to replace a bridge, when to switch a signal | [slides](lectures/build/cee_l26_reinforcement_learning.pdf), [pptx](lectures/build/cee_l26_reinforcement_learning.pptx), [notebook](ipynb/cee_l26_reinforcement_learning.ipynb) |
| 27 | Fairness and Accountability | Does Lecture 6's bridge model serve Illinois's vulnerable counties as well as the rest? | [slides](lectures/build/cee_l27_fairness.pdf), [pptx](lectures/build/cee_l27_fairness.pptx), [notebook](ipynb/cee_l27_fairness.ipynb) |
| 28 | AI as Infrastructure | A data centre asks to connect in Illinois: what does it draw, what does it emit, and who signs? | [slides](lectures/build/cee_l28_ai_infrastructure.pdf), [pptx](lectures/build/cee_l28_ai_infrastructure.pptx), [notebook](ipynb/cee_l28_ai_infrastructure.ipynb) |

## Quick start

```bash
# Install the pinned environment (one time, a few minutes)
pixi install -e aml-env

# Fetch the data for a lecture
bash scripts/fetch_concrete.sh

# Open a notebook in the reactive editor
pixi run -e aml-env marimo edit notebooks/cee_l02_nearest_neighbors.py

# Or run it as a script, so that every check executes
pixi run -e aml-env python notebooks/cee_l02_nearest_neighbors.py
```

The Jupyter exports in `ipynb/` open in any Jupyter environment. The environment file targets Linux
with an NVIDIA GPU. The notebooks choose the device at run time and also run on a CPU.

## Credits

The slides, figures, and code are my own. Some slides are inspired by Derek Hoiem's CS 441
*Applied Machine Learning* at the University of Illinois Urbana-Champaign. The primary textbook is
Forsyth, *Applied Machine Learning*, Springer, 2019 ([DOI](https://doi.org/10.1007/978-3-030-18114-7)).

## Licence

- Code (figure scripts, notebooks, tools, fetch scripts): MIT, see [`LICENSE`](LICENSE).
- Slides and text: Creative Commons Attribution 4.0, see [`LICENSE-CC-BY-4.0.txt`](LICENSE-CC-BY-4.0.txt).
- Third-party data keeps its own licence. Figures and images made from a dataset carry that
  dataset's terms, which each fetch script states. Four datasets add conditions beyond
  attribution: SpaceNet 8 and Building Data Genome 2 (CC BY-SA 4.0), the Portland I-205 loop
  detector data (CC BY-SA 3.0), and the KW51 railway bridge data (CC BY-NC-SA 4.0).

## Author

Wilfredo Torres Calderon, <https://wtorresc.github.io>
