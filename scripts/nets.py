"""DeepSurv (Katzman 2018) and FT-Transformer (Gorishniy 2021) in PyTorch, for the v4 learner library.

Implemented in-repository rather than taken from an external package so that the pre-registered
model list is reproducible from the pinned torch version alone."""
import numpy as np, torch, torch.nn as nn

def _dev():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, p, hidden=(64, 64), dropout=0.2):
        super().__init__()
        layers, d = [], p
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]; d = h
        layers += [nn.Linear(d, 1)]
        self.f = nn.Sequential(*layers)
    def forward(self, x): return self.f(x).squeeze(-1)

def cox_ph_loss(risk, time, event):
    """Negative Breslow partial log-likelihood, computed on the batch's risk set."""
    order = torch.argsort(time, descending=True)
    r, e = risk[order], event[order]
    logcum = torch.logcumsumexp(r, dim=0)
    ll = (r - logcum) * e
    return -ll.sum()/torch.clamp(e.sum(), min=1.0)

class DeepSurv:
    """MLP trained on the Cox partial likelihood; predicts S(t) via a Breslow baseline."""
    def __init__(self, hidden=(64, 64), dropout=0.2, lr=1e-3, epochs=60, batch=512, seed=20260911):
        self.kw = dict(hidden=hidden, dropout=dropout); self.lr, self.epochs, self.batch, self.seed = lr, epochs, batch, seed
    def fit(self, X, time, event):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.dev = _dev(); X = np.asarray(X, dtype=np.float32)
        self.net = MLP(X.shape[1], **self.kw).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        xt = torch.tensor(X, device=self.dev); tt = torch.tensor(np.asarray(time, dtype=np.float32), device=self.dev)
        et = torch.tensor(np.asarray(event, dtype=np.float32), device=self.dev)
        n = len(X)
        for ep in range(self.epochs):
            self.net.train(); perm = torch.randperm(n, device=self.dev)
            for i in range(0, n, self.batch):
                ix = perm[i:i+self.batch]
                if len(ix) < 16 or et[ix].sum() < 1: continue
                opt.zero_grad(); loss = cox_ph_loss(self.net(xt[ix]), tt[ix], et[ix]); loss.backward(); opt.step()
        self.net.eval()
        with torch.no_grad(): lp = self.net(xt).cpu().numpy()
        self._breslow(np.asarray(time, dtype=float), np.asarray(event, dtype=float), lp)
        return self
    def _breslow(self, time, event, lp):
        order = np.argsort(time); t, e, r = time[order], event[order], np.exp(lp[order])
        self.times, h = [], []
        cum = 0.0
        for i, ti in enumerate(t):
            if e[i] == 1:
                risk = r[i:].sum()
                cum += 1.0/max(risk, 1e-9); self.times.append(ti); h.append(cum)
        self.times, self.H0 = np.asarray(self.times), np.asarray(h)
    def predict_lp(self, X):
        with torch.no_grad():
            return self.net(torch.tensor(np.asarray(X, dtype=np.float32), device=self.dev)).cpu().numpy()
    def predict_surv(self, X, t):
        if not len(self.times): return np.ones(len(X))
        H = self.H0[np.searchsorted(self.times, t, "right")-1] if t >= self.times[0] else 0.0
        return np.exp(-H*np.exp(self.predict_lp(X)))

class FTTransformer(nn.Module):
    """Feature tokenizer plus transformer encoder over numeric features, with a CLS head."""
    def __init__(self, p, d=64, layers=3, heads=8, dropout=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(p, d)*0.02); self.b = nn.Parameter(torch.zeros(p, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d)*0.02)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=2*d, dropout=dropout, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))
    def forward(self, x):
        tok = x.unsqueeze(-1)*self.w.unsqueeze(0) + self.b.unsqueeze(0)
        z = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)
        return self.head(self.tr(z)[:, 0]).squeeze(-1)

class FTClassifier:
    def __init__(self, d=64, layers=3, heads=8, dropout=0.1, lr=1e-3, epochs=40, batch=512, seed=20260911):
        self.kw = dict(d=d, layers=layers, heads=heads, dropout=dropout); self.lr, self.epochs, self.batch, self.seed = lr, epochs, batch, seed
    def fit(self, X, y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.dev = _dev(); X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        self.net = FTTransformer(X.shape[1], **self.kw).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss()
        xt, yt = torch.tensor(X, device=self.dev), torch.tensor(y, device=self.dev)
        n = len(X)
        for ep in range(self.epochs):
            self.net.train(); perm = torch.randperm(n, device=self.dev)
            for i in range(0, n, self.batch):
                ix = perm[i:i+self.batch]
                if len(ix) < 16: continue
                opt.zero_grad(); loss = lossf(self.net(xt[ix]), yt[ix]); loss.backward(); opt.step()
        self.net.eval(); return self
    def predict_proba(self, X):
        with torch.no_grad():
            out = []
            X = np.asarray(X, dtype=np.float32)
            for i in range(0, len(X), 4096):
                out.append(torch.sigmoid(self.net(torch.tensor(X[i:i+4096], device=self.dev))).cpu().numpy())
        p = np.concatenate(out) if out else np.zeros(0)
        return np.c_[1-p, p]
