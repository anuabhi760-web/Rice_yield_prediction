# Rice Yield Project — Deployment (dashboard + chatbot)

## Files
| File | What it is |
|---|---|
| `app.py` | Streamlit app. Sidebar switch: **📈 Dashboard** / **💬 Chatbot** |
| `chatbot.py` | The chatbot (data-grounded answers + optional LLM layer). Must sit next to `app.py` |
| `01_train_final_models.py` | Trains Random Forest + XGBoost, SHAP, **and refits production models on all years** |
| `02_train_state_space_model.py` | Rolling one-step-ahead State Space Model + **next-season forecast with 95% interval** |
| `requirements.txt`, `secrets.toml.example` | Dependencies; template for the optional API key |

## Run order
```bash
pip install -r requirements.txt
python 01_train_final_models.py          # writes *.pkl, test_predictions.csv, model_summary.csv, shap_global_importance.csv
python 02_train_state_space_model.py     # writes ssm_forecasts.csv, ssm_next_year.csv, ssm_diagnostics.csv
streamlit run app.py
```
Upload to GitHub (same folder as `app.py`): `app.py`, `chatbot.py`, `requirements.txt`, all `*.pkl`, all `*.csv` the scripts wrote,
and the dataset (`rice_model_ready_full.csv` or `rice_dataset_final.csv`, flat or in a `data/` folder — both work).
Each `.pkl` is under 25 MB (the training script prints the sizes), so the normal browser upload works.

**Pin versions.** Pickled models are sensitive to library versions. Run `pip freeze` on the machine that trained them and pin
`scikit-learn`, `xgboost`, `joblib`, `streamlit` in `requirements.txt` to the same versions.

## Two sets of models — on purpose
* `random_forest.pkl`, `xgboost.pkl` — trained on **1966–2005**, so they can be scored honestly on 2006–2025. Used for the test-period charts.
* `random_forest_full.pkl`, `xgboost_full.pkl` — **refit on all years**. Used for next-season forecasts (chatbot + what-if panel).
  A tree model cannot predict above the highest yields it saw in training, so the 1966–2005 models under-predict today's yields
  (in testing: Patna 4,163 vs 4,915 kg/ha). If the `*_full` files are missing the chatbot still works but shows a warning.

## Chatbot
Ask things like: *What was the yield in Patna in 2010?* · *What will rice yield be in Patna this Kharif?* · *What if drought hits Guntur?* ·
*What if rainfall is 20% less in Gaya?* · *Compare Patna and Gaya* · *Top 5 districts in Punjab* · *Yield trend in West Bengal since 2000* ·
*What factors drive rice yield?* · *How accurate are the models?* · *How sure are you about Ludhiana?* · *Advice for farmers in Raichur*.
It remembers the last district (*"and 2015?"*, *"advice?"*), tolerates typos and alternate names (Trichy, Odisha, Vizag), and says so when a place isn't in the data.

**Layer 1 (always on, no key):** parses the question, looks the answer up in the data / saved model outputs, replies with a table and chart.
Every number comes from the dataset or the models, so it cannot invent figures.
**Layer 2 (optional):** with an API key, an LLM rewrites the same facts conversationally and handles Hindi/Telugu/Bengali/Tamil
(translate → look up → answer in the user's language). It is instructed to use only the supplied facts, and any failure falls back to layer 1.

Enable layer 2: Streamlit Cloud → *Settings → Secrets* → `ANTHROPIC_API_KEY = "..."` (see `secrets.toml.example`).
Never put a key in the code or the repo. Each browser session is capped at 30 LLM calls to protect your credit.
For a quick demo you can also paste a key into the sidebar (kept for that session only).

## Data-quality switch in `chatbot.py`
`DATA_TRUSTED_THROUGH = 2015`. Checks on the dataset found that from 2016 on, both climate and yield look smoothed/extrapolated rather than
observed (see the main write-up). The chatbot labels anything built on those years as indicative and computes its dry/hot-year analysis
only from years up to 2015. Once you have traced the source of the 2016–2025 rows, set it to `None` to switch the notes off.

## Not (yet) connected
No live weather forecast, no real-time prices, soil values for many districts are state-level estimates. Advisories are generic, rule-based
pointers — not prescriptions.
