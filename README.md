# NrityaVaani - Dataset

Project containing dataset and training scripts for dance posture classification.

## Image Count per Mudra (Train / Validation / Test)

### Training Set
| Mudra         | Image Count |
|--------------|-------------|
| Alapadma     | 194 |
| Ardhapataka  | 189 |
| Chandrakala  | 186 |
| Kartarimukha | 212 |
| Mayura       | 184 |
| Pataka       | 183 |
| Shikhara     | 188 |
| Simhamukha   | 189 |
| Suchi        | 186 |
| Tripataka    | 189 |

### Validation Set
| Mudra         | Image Count |
|--------------|-------------|
| Alapadma     | 8 |
| Ardhapataka  | 8 |
| Chandrakala  | 8 |
| Kartarimukha | 11 |
| Mayura       | 7 |
| Pataka       | 11 |
| Shikhara     | 9 |
| Simhamukha   | 9 |
| Suchi        | 9 |
| Tripataka    | 6 |

### Test Set
| Mudra         | Image Count |
|--------------|-------------|
| Alapadma     | 9 |
| Ardhapataka  | 9 |
| Chandrakala  | 9 |
| Kartarimukha | 12 |
| Mayura       | 9 |
| Pataka       | 14 |
| Shikhara     | 11 |
| Simhamukha   | 9 |
| Suchi        | 11 |
| Tripataka    | 6 |


Contents:

- `cvModel.py`, `train.py`, model weights `nrityavaani_mobilenet.pth`
- dataset in `final_dataset/` (train/val/test splits)

To initialize and push this repo (example):

1. `git init`
2. `git add -A`
3. `git commit -m "Initial commit"`
4. Create a remote repo on GitHub and run:
   `git remote add origin <URL>`
   `git branch -M main`
   `git push -u origin main`
