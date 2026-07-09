import json
import numpy as np
from src.config import PSO_PARTICLES, PSO_ITERS, ENSEMBLE_CFG


class PSO:
    def __init__(self, objective_fn, n_particles=PSO_PARTICLES, n_dims=2,
                 bounds=(0.0, 1.0), n_iter=PSO_ITERS,
                 c1=1.5, c2=1.5, w=0.40):
        self.fn          = objective_fn
        self.n_p         = n_particles
        self.n_d         = n_dims
        self.lo, self.hi = bounds
        self.n_iter      = n_iter
        self.c1 = c1; self.c2 = c2; self.w = w

        rng      = np.random.default_rng(42)
        self.pos = rng.uniform(self.lo, self.hi, (n_particles, n_dims))
        v_max    = (self.hi - self.lo) * 0.15
        self.vel = rng.uniform(-v_max, v_max, (n_particles, n_dims))

        self.pbest_pos   = self.pos.copy()
        self.pbest_score = np.array([self._eval(p) for p in self.pos])
        gi               = self.pbest_score.argmax()
        self.gbest_pos   = self.pbest_pos[gi].copy()
        self.gbest_score = self.pbest_score[gi]
        self.history     = [self.gbest_score]

    def _eval(self, pos):
        w = np.abs(pos) + 1e-9
        w = w / w.sum()
        return self.fn(w)

    def run(self, verbose=True):
        rng = np.random.default_rng(0)
        for it in range(self.n_iter):
            r1 = rng.random((self.n_p, self.n_d))
            r2 = rng.random((self.n_p, self.n_d))
            self.vel = (self.w * self.vel
                        + self.c1 * r1 * (self.pbest_pos - self.pos)
                        + self.c2 * r2 * (self.gbest_pos - self.pos))
            self.pos = np.clip(self.pos + self.vel, self.lo, self.hi)

            scores = np.array([self._eval(p) for p in self.pos])
            improved = scores > self.pbest_score
            self.pbest_pos[improved]   = self.pos[improved]
            self.pbest_score[improved] = scores[improved]
            if scores.max() > self.gbest_score:
                self.gbest_score = scores.max()
                self.gbest_pos   = self.pos[scores.argmax()].copy()
            self.history.append(self.gbest_score)

            if verbose and (it + 1) % 10 == 0:
                w = np.abs(self.gbest_pos); w /= w.sum()
                print(f"  PSO iter {it+1:3d}/{self.n_iter}  "
                      f"best_val_acc={self.gbest_score:.5f}  "
                      f"w_eff={w[0]:.3f}  w_vit={w[1]:.3f}")

        best_w = np.abs(self.gbest_pos)
        best_w = best_w / best_w.sum()
        return best_w, self.history


def find_ensemble_weights(eff_val_probs, vit_val_probs, val_true):
    eff_solo = (eff_val_probs.argmax(1) == val_true).mean()
    vit_solo = (vit_val_probs.argmax(1) == val_true).mean()
    naive    = ((0.5*eff_val_probs + 0.5*vit_val_probs).argmax(1) == val_true).mean()
    print(f"EfficientNetB7 val acc : {eff_solo:.4f}")
    print(f"ViT-B16        val acc : {vit_solo:.4f}")
    print(f"Naive 50/50    val acc : {naive:.4f}")

    def objective(w):
        return ((w[0]*eff_val_probs + w[1]*vit_val_probs).argmax(1) == val_true).mean()

    pso = PSO(objective)
    best_w, history = pso.run(verbose=True)
    print(f"PSO result: w_eff={best_w[0]:.4f}  w_vit={best_w[1]:.4f}  "
          f"val_acc={history[-1]:.4f}")
    return best_w, history


def ensemble_predict(eff_probs, vit_probs, weights):
    return weights[0] * eff_probs + weights[1] * vit_probs


def save_ensemble_config(weights, classes, val_acc, test_acc):
    cfg = {
        'eff_weight':    float(weights[0]),
        'vit_weight':    float(weights[1]),
        'classes':       classes,
        'val_accuracy':  float(val_acc),
        'test_accuracy': float(test_acc),
    }
    with open(ENSEMBLE_CFG, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved ensemble config: {ENSEMBLE_CFG}")
    return cfg


def load_ensemble_config():
    with open(ENSEMBLE_CFG) as f:
        return json.load(f)
