import os
import numpy as np

root = os.path.join(os.getcwd(), 'checkpoint')
models = {'Backbone_CNN': [], 'TFN_Gabor': []}

if not os.path.isdir(root):
    print('NO_CHECKPOINT_DIR')
    raise SystemExit(1)

for name in os.listdir(root):
    path = os.path.join(root, name)
    if not os.path.isdir(path):
        continue
    labf = os.path.join(path, 'test_labels.npy')
    predf = os.path.join(path, 'test_preds.npy')
    if not (os.path.isfile(labf) and os.path.isfile(predf)):
        continue
    try:
        labels = np.load(labf)
        preds = np.load(predf)
    except Exception as e:
        print(f'LOAD_ERROR {name} {e}')
        continue
    if preds.ndim > 1:
        try:
            pred_labels = np.argmax(preds, axis=1)
        except Exception:
            pred_labels = preds.ravel()
    else:
        pred_labels = preds.ravel()
    labels = labels.ravel()
    if labels.shape[0] != pred_labels.shape[0]:
        print(f'SHAPE_MISMATCH {name} labels:{labels.shape} preds:{pred_labels.shape}')
        continue
    acc = float((pred_labels == labels).astype(float).mean())
    if 'Backbone_CNN' in name:
        models['Backbone_CNN'].append((name, acc))
    if 'TFN_Gabor' in name:
        models['TFN_Gabor'].append((name, acc))

# Print per-run and aggregate
for k, v in models.items():
    if not v:
        print(f'{k} NO_RUNS')
        continue
    print(f'Model {k} runs:')
    for n, a in v:
        print(f'  {n}: {a:.6f}')
    mean_acc = sum(a for _, a in v) / len(v)
    print(f'Mean {k} accuracy across {len(v)} runs: {mean_acc:.6f}')

# Summary line for easy parsing
for k, v in models.items():
    if not v:
        print(f'SUMMARY {k} NaN')
    else:
        mean_acc = sum(a for _, a in v) / len(v)
        print(f'SUMMARY {k} {mean_acc:.6f}')
