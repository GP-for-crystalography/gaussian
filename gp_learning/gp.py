import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# Przeniesienie obliczeń GP na kartę graficzną
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_x i train_y na GPU
train_x = train_x.to(device)
train_y = train_y.to(device)

gp = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# Generowanie paczki 4 punktów naraz (q=4)
qEI = qExpectedImprovement(gp, best_f=train_y.max())
candidates, _ = optimize_acqf(
    qEI, 
    bounds=bounds.to(device), 
    q=4, # <--- liczba punktów do zbadania równolegle
    num_restarts=10, 
    raw_samples=512
)
# candidates to tensor 4 punktów gotowych do równoległego uruchomienia np. przez multiprocessing