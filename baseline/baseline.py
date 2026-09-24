import os
import math
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())

# ─── CHANGE THESE ─────────────────────────────────────────────────────
DATASET = "chicago_beats"
SEED    = 42
# ──────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
LOOK_BACK   = 4

CONFIGS = {
    "chicago_beats": {
        "crime"  : "data/processed/chicago_beats_crime_timeseries.csv",
        "order"  : "data/raw/graph_beat_order.csv",
        "adj_fc" : "data/processed/chicago_beats_adjacencyfc.npy",
    },
    "chicago_districts": {
        "crime"  : "data/processed/chicago_districts_crime_timeseries.csv",
        "order"  : "data/raw/graph_district_order.csv",
        "adj_fc" : "data/processed/chicago_districts_adjacencyfc.npy",
    },
    "nyc_precincts": {
        "crime"  : "data/processed/nyc_precincts_crime_timeseries.csv",
        "order"  : "data/raw/graph_precinct_order.csv",
        "adj_fc" : "data/processed/nyc_precincts_adjacencyfc.npy",
    },
}
cfg = CONFIGS[DATASET]

# ═════════════════════════════════════════════════════════════════════
# 1. DATA
# ═════════════════════════════════════════════════════════════════════
crime        = pd.read_csv(cfg["crime"])
region_order = pd.read_csv(cfg["order"]).iloc[:, 0].astype(int).tolist()
regions      = region_order
months       = sorted(crime["month"].unique())
N            = len(regions)
T            = len(months)

C_all = np.zeros((T, N), dtype=np.float32)
for i, m in enumerate(months):
    c = (crime[crime["month"] == m]
         .set_index("region_id").reindex(regions)
         .fillna(0)["C"].values)
    C_all[i] = c

n_train      = int(T * TRAIN_RATIO)
n_val        = int(T * VAL_RATIO)
train_idx    = list(range(0, n_train))
val_idx      = list(range(n_train, n_train + n_val))
test_idx     = list(range(n_train + n_val, T))
train_C      = C_all[train_idx]
val_C        = C_all[val_idx]
test_C       = C_all[test_idx]
train_val_C  = np.vstack([train_C, val_C])

A_np  = np.load(cfg["adj_fc"]).astype(np.float32)
A_row = A_np / A_np.sum(axis=1, keepdims=True).clip(min=1.0)

print(f"Dataset={DATASET}  N={N}  T={T}  Seed={SEED}")
print(f"Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════════════
# 2. METRICS  (sMAPE only — MAPE excluded for sparse crime data)
# ═════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, name):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    smape = float(np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)
    print(f"  {name:<22} MAE={mae:.4f}  RMSE={rmse:.4f}  "
          f"R2={r2:.4f}  sMAPE={smape:.2f}%")
    return dict(Model=name, MAE=mae, RMSE=rmse, R2=r2, sMAPE=smape)

results      = []
error_arrays = {}

def record(name, y_pred_2d):
    y_true = test_C.flatten()
    y_pred = np.array(y_pred_2d, dtype=np.float32).flatten()
    m      = compute_metrics(y_true, y_pred, name)
    results.append(m)
    error_arrays[name] = (y_pred - y_true) ** 2

# ═════════════════════════════════════════════════════════════════════
# 3. HELPERS
# ═════════════════════════════════════════════════════════════════════
def make_sequences(data, look_back):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        Y.append(data[i + look_back])
    return (np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32))


def autoregressive_predict(model_fn, seed_data, n_steps, look_back):
    buf   = list(seed_data[-look_back:])
    preds = []
    for _ in range(n_steps):
        fc = model_fn(np.array(buf[-look_back:], dtype=np.float32))
        preds.append(np.array(fc, dtype=np.float32))
        buf.append(fc)
    return np.array(preds, dtype=np.float32)


def train_nn(model, X_tr, Y_tr, X_val, Y_val,
             lr=1e-3, epochs=200, patience=20, batch=16):
    opt     = torch.optim.Adam(model.parameters(),
                                lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=8, factor=0.5, min_lr=1e-6)
    loss_fn = nn.HuberLoss(delta=0.3)
    Xt = torch.tensor(X_tr)
    Yt = torch.tensor(Y_tr)
    Xv = torch.tensor(X_val)
    Yv = torch.tensor(Y_val)
    best_val = float("inf")
    best_w   = None
    pat_c    = 0

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            b = idx[i:i + batch]
            opt.zero_grad()
            loss_fn(model(Xt[b]), Yt[b]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), Yv).item()
        sched.step(vl)
        if vl < best_val:
            best_val = vl
            best_w   = {k: v.clone()
                        for k, v in model.state_dict().items()}
            pat_c    = 0
        else:
            pat_c += 1
            if pat_c >= patience:
                break

    if best_w:
        model.load_state_dict(best_w)
    return model


X_tv, Y_tv = make_sequences(train_val_C, LOOK_BACK)
X_va, Y_va = make_sequences(val_C,       LOOK_BACK)

# ═════════════════════════════════════════════════════════════════════
# 4. MODEL DEFINITIONS
# ═════════════════════════════════════════════════════════════════════

# ── LSTM ──────────────────────────────────────────────────────────────
class SimpleLSTM(nn.Module):
    def __init__(self, n, hidden=64, layers=2):
        super().__init__()
        self.rnn = nn.LSTM(n, hidden, layers, batch_first=True,
                           dropout=0.1 if layers > 1 else 0.0)
        self.fc  = nn.Linear(hidden, n)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


# ── GRU ───────────────────────────────────────────────────────────────
class SimpleGRU(nn.Module):
    def __init__(self, n, hidden=64, layers=2):
        super().__init__()
        self.rnn = nn.GRU(n, hidden, layers, batch_first=True,
                          dropout=0.1 if layers > 1 else 0.0)
        self.fc  = nn.Linear(hidden, n)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


# ── TCN ───────────────────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, d):
        super().__init__()
        self.pad  = (k - 1) * d
        self.conv = nn.Conv1d(in_ch, out_ch, k,
                              dilation=d, padding=self.pad)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, k, d), nn.ReLU(),
            CausalConv1d(out_ch, out_ch, k, d), nn.ReLU(),
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) \
                   if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.res(x)


class TCN(nn.Module):
    def __init__(self, n, hidden=64, n_layers=4):
        super().__init__()
        layers = []
        ch     = n
        for i in range(n_layers):
            layers.append(TCNBlock(ch, hidden, k=3, d=2**i))
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.fc  = nn.Linear(hidden, n)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.fc(self.net(x)[:, :, -1])


# ── Transformer ───────────────────────────────────────────────────────
class TransformerForecaster(nn.Module):
    def __init__(self, n, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(n, d_model)
        enc       = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4,
            dropout=0.1, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc, layers)
        self.fc   = nn.Linear(d_model, n)

    def forward(self, x):
        return self.fc(self.enc(self.proj(x))[:, -1])


# ── DCRNN ─────────────────────────────────────────────────────────────
class DCRNNCell(nn.Module):
    def __init__(self, N, in_d, hidden, k=2):
        super().__init__()
        self.N      = N
        self.hidden = hidden
        feat        = (in_d + hidden) * (k + 1)
        self.gate   = nn.Linear(feat, 2 * hidden)
        self.upd    = nn.Linear(feat, hidden)

    def _diff(self, x, Als):
        out = [x]
        for A in Als:
            out.append(torch.einsum("ij,bjf->bif", A, out[-1]))
        return torch.cat(out, dim=-1)

    def forward(self, x, h, Als):
        B   = x.size(0)
        xh  = torch.cat([x, h], dim=-1)
        d   = self._diff(xh, Als).reshape(B * self.N, -1)
        g   = torch.sigmoid(self.gate(d))
        r, u = g.split(self.hidden, dim=1)
        xr  = torch.cat([x.reshape(B * self.N, -1),
                          r * h.reshape(B * self.N, -1)], dim=1)
        xrd = self._diff(
            xr.reshape(B, self.N, -1), Als
        ).reshape(B * self.N, -1)
        c   = torch.tanh(self.upd(xrd))
        h_n = u * h.reshape(B * self.N, -1) + (1 - u) * c
        return h_n.reshape(B, self.N, self.hidden)


class DCRNN(nn.Module):
    def __init__(self, N, A, hidden=32, layers=2):
        super().__init__()
        At  = torch.tensor(A, dtype=torch.float32)
        fw  = At / At.sum(1, keepdim=True).clamp(min=1)
        bw  = At.T / At.T.sum(1, keepdim=True).clamp(min=1)
        self.register_buffer("fw", fw)
        self.register_buffer("bw", bw)
        self.cells  = nn.ModuleList([
            DCRNNCell(N, 1 if i == 0 else hidden, hidden)
            for i in range(layers)
        ])
        self.hidden = hidden
        self.N      = N
        self.fc     = nn.Linear(hidden, 1)

    def forward(self, x):
        B, L, N_ = x.shape
        Als      = [self.fw, self.bw]
        hs       = [torch.zeros(B, N_, self.hidden, device=x.device)
                    for _ in self.cells]
        for t in range(L):
            inp = x[:, t].unsqueeze(-1)
            for i, cell in enumerate(self.cells):
                inp    = cell(inp, hs[i], Als)
                hs[i]  = inp
        return self.fc(hs[-1]).squeeze(-1)


# ── STGCN ─────────────────────────────────────────────────────────────
class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A_hat):
        super().__init__()
        self.register_buffer("A", A_hat)
        self.theta = nn.Linear(in_ch, out_ch, bias=False)
        self.tc1   = nn.Conv2d(1, out_ch, (1, 3), padding=(0, 1))
        self.tc2   = nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, 1))
        self.bn    = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        B, C, N_, T_ = x.shape
        xt   = x.permute(0, 3, 2, 1).reshape(B * T_, N_, C)
        gc   = torch.einsum("ij,bjf->bif", self.A, self.theta(xt))
        gc   = gc.reshape(B, T_, N_, -1).permute(0, 3, 2, 1)
        t1   = torch.relu(
            self.tc1(gc[:, :1].permute(0, 1, 3, 2)).permute(0, 1, 3, 2))
        t2   = torch.relu(
            self.tc2(t1[:, :1].permute(0, 1, 3, 2)).permute(0, 1, 3, 2))
        return self.bn(t2)


class STGCN(nn.Module):
    def __init__(self, N, A, look_back, hidden=32):
        super().__init__()
        At   = torch.tensor(A, dtype=torch.float32)
        D    = torch.diag(At.sum(1).clamp(min=1).pow(-0.5))
        Ah   = D @ At @ D
        self.b1  = STGCNBlock(1, hidden, Ah)
        self.b2  = STGCNBlock(hidden, hidden, Ah)
        self.fc  = nn.Linear(hidden * look_back, N)
        self.N   = N
        self.lb  = look_back

    def forward(self, x):
        B, L, N_ = x.shape
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.b1(x)
        x = self.b2(x)
        x = x.reshape(B, -1, N_).permute(0, 2, 1).reshape(B * N_, -1)
        return self.fc(x).mean(0).unsqueeze(0).expand(B, -1)


# ── Graph WaveNet ─────────────────────────────────────────────────────
class WaveNetLayer(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.fconv = nn.Conv1d(ch, ch, 2, dilation=dilation,
                               padding=dilation)
        self.gconv = nn.Conv1d(ch, ch, 2, dilation=dilation,
                               padding=dilation)
        self.res   = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        T_  = x.size(-1)
        f   = torch.tanh(self.fconv(x)[:, :, :T_])
        g   = torch.sigmoid(self.gconv(x)[:, :, :T_])
        return self.res(x) + f * g


class GraphWaveNet(nn.Module):
    def __init__(self, N, look_back, hidden=32):
        super().__init__()
        self.start  = nn.Linear(N, hidden)
        self.layers = nn.ModuleList([
            WaveNetLayer(hidden, 2**i) for i in range(4)
        ])
        self.end    = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden * look_back, N),
        )
        self.lb = look_back

    def forward(self, x):
        B, L, N_ = x.shape
        x = self.start(x).permute(0, 2, 1)
        for lyr in self.layers:
            x = lyr(x)
        return self.end(x.reshape(B, -1))


# ── AGCRN ─────────────────────────────────────────────────────────────
class AGCRNCell(nn.Module):
    def __init__(self, N, emb=16, hidden=32):
        super().__init__()
        self.N      = N
        self.hidden = hidden
        self.E      = nn.Parameter(torch.randn(N, emb) * 0.01)
        in_d        = hidden + 1
        self.gate   = nn.Linear(in_d * 2, 2 * hidden)
        self.upd    = nn.Linear(in_d * 2, hidden)

    def _A(self):
        return torch.softmax(torch.relu(self.E @ self.E.T), dim=1)

    def forward(self, x, h):
        B   = x.size(0)
        A   = self._A()
        xh  = torch.cat([x, h], dim=-1)
        agg = torch.einsum("ij,bjf->bif", A, xh)
        cat = torch.cat([agg, xh], dim=-1).reshape(B * self.N, -1)
        g   = torch.sigmoid(self.gate(cat))
        r, u = g.split(self.hidden, dim=1)
        h_  = h.reshape(B * self.N, self.hidden)
        xr  = torch.cat([x.reshape(B * self.N, -1), r * h_], dim=1)
        xr2 = torch.cat([xr, xr], dim=1)
        c   = torch.tanh(self.upd(xr2))
        h_n = u * h_ + (1 - u) * c
        return h_n.reshape(B, self.N, self.hidden)


class AGCRN(nn.Module):
    def __init__(self, N, emb=16, hidden=32, layers=2):
        super().__init__()
        self.cells  = nn.ModuleList([
            AGCRNCell(N, emb, hidden) for _ in range(layers)
        ])
        self.hidden = hidden
        self.N      = N
        self.fc     = nn.Linear(hidden, 1)

    def forward(self, x):
        B, L, N_ = x.shape
        hs = [torch.zeros(B, N_, self.hidden, device=x.device)
              for _ in self.cells]
        for t in range(L):
            inp = x[:, t].unsqueeze(-1)
            for i, cell in enumerate(self.cells):
                inp    = cell(inp, hs[i])
                hs[i]  = inp
        return self.fc(hs[-1]).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════
# 5. BASELINES
# ═════════════════════════════════════════════════════════════════════
print("\nRunning baselines ...")

# ── Historical Mean ───────────────────────────────────────────────────
record("HistMean",
       np.tile(train_val_C.mean(axis=0), (len(test_idx), 1)))

# ── Naïve / Persistence ───────────────────────────────────────────────
naive = np.vstack(
    [train_val_C[-1:]] +
    [test_C[i:i+1] for i in range(len(test_idx) - 1)]
)
record("Naive", naive)

# ── VAR with PCA fallback ─────────────────────────────────────────────
def run_var(data, n_test):
    try:
        mv  = VAR(data)
        lag = max(mv.select_order(maxlags=4)
                  .selected_orders.get("aic", 1), 1)
        res = mv.fit(lag, trend="c")
        buf = list(data)
        out = []
        for _ in range(n_test):
            fc = res.forecast(np.array(buf[-lag:]), steps=1)[0]
            out.append(fc); buf.append(fc)
        return np.array(out), f"VAR(lag={lag})", None
    except Exception as e:
        return None, None, str(e)

preds_var, var_label, var_err = run_var(train_val_C, len(test_idx))
if preds_var is not None:
    record(var_label, preds_var)
    print(f"    {var_label}")
else:
    print(f"    VAR failed ({var_err}) — falling back to VAR+PCA(20)")
    pca      = PCA(n_components=min(20, N-1)).fit(train_val_C)
    tv_pca   = pca.transform(train_val_C)
    preds_v, pca_label, pca_err = run_var(tv_pca, len(test_idx))
    if preds_v is not None:
        record("VAR+PCA(20)",
               pca.inverse_transform(preds_v))
        print(f"    VAR+PCA(20) done")
    else:
        print(f"    VAR+PCA also failed: {pca_err}")

# ── Ridge Regression ─────────────────────────────────────────────────
ridge = Ridge(alpha=1.0).fit(train_val_C[:-1], train_val_C[1:])
record("Ridge",
       autoregressive_predict(
           lambda x: ridge.predict(
               x[-1:].reshape(1, -1))[0],
           train_val_C, len(test_idx), 1))

# ── LSTM ──────────────────────────────────────────────────────────────
t0   = time.time()
lstm = SimpleLSTM(N, 64, 2)
train_nn(lstm, X_tv, Y_tv, X_va, Y_va)
lstm.eval()
with torch.no_grad():
    preds_lstm = autoregressive_predict(
        lambda x: lstm(
            torch.tensor(x).unsqueeze(0)
        ).squeeze(0).numpy(),
        train_val_C, len(test_idx), LOOK_BACK)
record("LSTM", preds_lstm)
print(f"    LSTM  {time.time()-t0:.1f}s")

# ── GRU ───────────────────────────────────────────────────────────────
t0  = time.time()
gru = SimpleGRU(N, 64, 2)
train_nn(gru, X_tv, Y_tv, X_va, Y_va)
gru.eval()
with torch.no_grad():
    preds_gru = autoregressive_predict(
        lambda x: gru(
            torch.tensor(x).unsqueeze(0)
        ).squeeze(0).numpy(),
        train_val_C, len(test_idx), LOOK_BACK)
record("GRU", preds_gru)
print(f"    GRU   {time.time()-t0:.1f}s")

# ── TCN ───────────────────────────────────────────────────────────────
t0  = time.time()
tcn = TCN(N, 64, 4)
train_nn(tcn, X_tv, Y_tv, X_va, Y_va)
tcn.eval()
with torch.no_grad():
    preds_tcn = autoregressive_predict(
        lambda x: tcn(
            torch.tensor(x).unsqueeze(0)
        ).squeeze(0).numpy(),
        train_val_C, len(test_idx), LOOK_BACK)
record("TCN", preds_tcn)
print(f"    TCN   {time.time()-t0:.1f}s")

# ── Transformer ───────────────────────────────────────────────────────
t0  = time.time()
tfm = TransformerForecaster(N, 64, 4, 2)
train_nn(tfm, X_tv, Y_tv, X_va, Y_va, lr=5e-4)
tfm.eval()
with torch.no_grad():
    preds_tfm = autoregressive_predict(
        lambda x: tfm(
            torch.tensor(x).unsqueeze(0)
        ).squeeze(0).numpy(),
        train_val_C, len(test_idx), LOOK_BACK)
record("Transformer", preds_tfm)
print(f"    Transformer  {time.time()-t0:.1f}s")

# ── DCRNN ─────────────────────────────────────────────────────────────
t0    = time.time()
dcrnn = DCRNN(N, A_row, 32, 2)
train_nn(dcrnn, X_tv, Y_tv, X_va, Y_va, lr=5e-4)
dcrnn.eval()
with torch.no_grad():
    preds_dcrnn = autoregressive_predict(
        lambda x: dcrnn(
            torch.tensor(x).unsqueeze(0)
        ).squeeze(0).numpy(),
        train_val_C, len(test_idx), LOOK_BACK)
record("DCRNN", preds_dcrnn)
print(f"    DCRNN  {time.time()-t0:.1f}s")

class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A_hat):
        super().__init__()
        self.register_buffer("A", A_hat)
        self.theta = nn.Linear(in_ch, out_ch, bias=False)
        self.tc1   = nn.Conv2d(in_ch,  out_ch, (1, 3), padding=(0, 1))
        self.tc2   = nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, 1))
        self.bn    = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # x: (B, C, N, T)
        B, C, N_, T_ = x.shape

        # graph convolution
        xt  = x.permute(0, 3, 2, 1).reshape(B * T_, N_, C)   # (B*T, N, C)
        gc  = torch.einsum("ij,bjf->bif", self.A, self.theta(xt))
        gc  = gc.reshape(B, T_, N_, -1).permute(0, 3, 2, 1)  # (B, out_ch, N, T)

        # temporal convolutions — operate on full channel dim, not sliced
        t1  = torch.relu(self.tc1(gc))   # (B, out_ch, N, T)
        t2  = torch.relu(self.tc2(t1))   # (B, out_ch, N, T)
        return self.bn(t2)


class STGCN(nn.Module):
    def __init__(self, N, A, look_back, hidden=32):
        super().__init__()
        At   = torch.tensor(A, dtype=torch.float32)
        D    = torch.diag(At.sum(1).clamp(min=1).pow(-0.5))
        Ah   = D @ At @ D
        self.b1  = STGCNBlock(1,      hidden, Ah)
        self.b2  = STGCNBlock(hidden, hidden, Ah)
        self.fc  = nn.Linear(hidden * look_back, N)
        self.N   = N
        self.lb  = look_back

    def forward(self, x):
        # x: (B, L, N)
        B, L, N_ = x.shape
        x = x.permute(0, 2, 1).unsqueeze(1)   # (B, 1, N, L)
        x = self.b1(x)                          # (B, hidden, N, L)
        x = self.b2(x)                          # (B, hidden, N, L)
        # pool over nodes, flatten time+channel -> predict all nodes
        x = x.mean(dim=2)                       # (B, hidden, L)
        x = x.reshape(B, -1)                    # (B, hidden*L)
        return self.fc(x)                        # (B, N)

# ── Graph WaveNet ─────────────────────────────────────────────────────
t0  = time.time()
gwn = GraphWaveNet(N, LOOK_BACK, 32)
train_nn(gwn, X_tv, Y_tv, X_va, Y_va, lr=5e-4)
gwn.eval()
with torch.no_grad():
    preds_gwn = autoregressive_predict(
        lambda x: gwn(
            torch.tensor(x).unsqueeze(0)
        ).squeeze(0).numpy(),
        train_val_C, len(test_idx), LOOK_BACK)
record("GraphWaveNet", preds_gwn)
print(f"    GraphWaveNet  {time.time()-t0:.1f}s")

class AGCRNCell(nn.Module):
    def __init__(self, N, emb=16, hidden=32):
        super().__init__()
        self.N      = N
        self.hidden = hidden
        self.E      = nn.Parameter(torch.randn(N, emb) * 0.01)
        # input dim: x(1) + h(hidden) = 1 + hidden, after graph agg same
        # concat [agg, original] = 2 * (1 + hidden)
        in_d        = 2 * (1 + hidden)
        self.gate   = nn.Linear(in_d, 2 * hidden)
        self.upd    = nn.Linear(in_d, hidden)

    def _A(self):
        return torch.softmax(torch.relu(self.E @ self.E.T), dim=1)

    def forward(self, x, h):
        # x: (B, N, 1)   h: (B, N, hidden)
        B   = x.size(0)
        A   = self._A()                              # (N, N)
        xh  = torch.cat([x, h], dim=-1)             # (B, N, 1+hidden)
        agg = torch.einsum("ij,bjf->bif", A, xh)    # (B, N, 1+hidden)
        cat = torch.cat([agg, xh], dim=-1)           # (B, N, 2*(1+hidden))
        cat = cat.reshape(B * self.N, -1)            # (B*N, 2*(1+hidden))
        g   = torch.sigmoid(self.gate(cat))          # (B*N, 2*hidden)
        r, u = g.split(self.hidden, dim=1)           # each (B*N, hidden)
        h_  = h.reshape(B * self.N, self.hidden)     # (B*N, hidden)
        # candidate
        xh_r = torch.cat([
            x.reshape(B * self.N, 1),
            r * h_
        ], dim=1)                                    # (B*N, 1+hidden)
        agg_r = torch.einsum(
            "ij,bjf->bif", A,
            xh_r.reshape(B, self.N, -1)
        ).reshape(B * self.N, -1)                   # (B*N, 1+hidden)
        cand_in = torch.cat([agg_r, xh_r], dim=1)  # (B*N, 2*(1+hidden))
        c   = torch.tanh(self.upd(cand_in))         # (B*N, hidden)
        h_n = u * h_ + (1 - u) * c                  # (B*N, hidden)
        return h_n.reshape(B, self.N, self.hidden)


class AGCRN(nn.Module):
    def __init__(self, N, emb=16, hidden=32, layers=2):
        super().__init__()
        self.cells  = nn.ModuleList([
            AGCRNCell(N, emb, hidden) for _ in range(layers)
        ])
        self.hidden = hidden
        self.N      = N
        self.fc     = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, L, N)
        B, L, N_ = x.shape
        hs = [torch.zeros(B, N_, self.hidden, device=x.device)
              for _ in self.cells]
        for t in range(L):
            inp = x[:, t].unsqueeze(-1)   # (B, N, 1)
            for i, cell in enumerate(self.cells):
                inp    = cell(inp, hs[i])
                hs[i]  = inp
        return self.fc(hs[-1]).squeeze(-1)   # (B, N)

# ═════════════════════════════════════════════════════════════════════
# 6. DIEBOLD-MARIANO TEST
# ═════════════════════════════════════════════════════════════════════
def diebold_mariano(e1, e2, h=1):
    d     = e1 - e2
    d_bar = d.mean()
    T_    = len(d)
    gamma = [np.var(d, ddof=0)]
    for k in range(1, h + 1):
        gamma.append(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T_
    DM    = d_bar / np.sqrt(max(var_d, 1e-10))
    p     = 2 * (1 - stats.norm.cdf(abs(DM)))
    return float(DM), float(p)

# ═════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ═════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"RESULTS — {DATASET.upper()}  seed={SEED}")
print("=" * 65)

df = pd.DataFrame(results).set_index("Model")
print(df.round(4).to_string())

best_name = df["MAE"].idxmin()
best_err  = error_arrays[best_name]
print(f"\nBest baseline: {best_name}  "
      f"MAE={df.loc[best_name,'MAE']:.4f}")

print("\nDiebold-Mariano test (all models vs best baseline):")
print(f"  {'Model':<22} {'DM':>8} {'p-val':>8} {'sig':>6}")
print("  " + "-" * 48)
for name, errs in error_arrays.items():
    if name == best_name:
        continue
    dm, p = diebold_mariano(errs, best_err)
    sig   = "✓" if p < 0.05 else "✗"
    print(f"  {name:<22} {dm:>8.3f} {p:>8.4f} {sig:>6}")

# ═════════════════════════════════════════════════════════════════════
# 8. PARAMETER COUNT
# ═════════════════════════════════════════════════════════════════════
print("\nParameter counts:")
named = [("LSTM", lstm), ("GRU", gru), ("TCN", tcn),
         ("Transformer", tfm), ("DCRNN", dcrnn),
         ("GraphWaveNet", gwn)]
for name, m in named:
    p = sum(x.numel() for x in m.parameters() if x.requires_grad)
    print(f"  {name:<22} {p:>10,}")

# ═════════════════════════════════════════════════════════════════════
# 9. SAVE
# ═════════════════════════════════════════════════════════════════════
os.makedirs("data/processed", exist_ok=True)
out = f"data/processed/{DATASET}_baselines_seed{SEED}.csv"
df.to_csv(out)
print(f"\nSaved -> {out}")
print("Done.")