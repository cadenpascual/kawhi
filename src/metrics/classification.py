from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss

def eval_probs(name, y_true, p):
    print(f"\n{name}")
    print("  log loss:", log_loss(y_true, p))
    print("  brier   :", brier_score_loss(y_true, p))