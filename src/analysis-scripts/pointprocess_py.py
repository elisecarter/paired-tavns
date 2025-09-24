# pointprocess_py.py
import numpy as np
from dataclasses import dataclass
from scipy.stats import invgauss
from scipy.optimize import minimize

# ------- public enum-like -------
class Distributions:
    InverseGaussian = "IG"

# ------- results -------
@dataclass
class RegressionResult:
    thetap: np.ndarray        # AR coefficients (theta_1..theta_p)
    theta0: float             # intercept (theta_0) if has_theta0, else 0
    kappa: float              # lambda (IG scale parameter)
    likelihood: float         # local log-likelihood at optimum
    mean_interval: float      # mu at window right-edge
    sigma: float              # sqrt(mu^3 / lambda)

@dataclass
class Result:
    # time series from compute_full_regression
    Time: np.ndarray
    Mu: np.ndarray
    lambda_: np.ndarray
    sd_RR: np.ndarray
    thetas: list  # list of (theta0, thetap) per window
    taus: np.ndarray  # for KS plotting if you compute residuals later

    def to_dict(self):
        return {"Time": self.Time, "Mu": self.Mu, "lambda": self.lambda_, "sd_RR": self.sd_RR}

# ------- helpers -------
def _window_design(events, ar_order, t_right, window_length, alpha, has_theta0):
    # events: beat times (s)
    mask = (events > t_right - window_length) & (events <= t_right)
    wbeats = events[mask]
    if wbeats.size < ar_order + 2:   # need at least p+1 intervals
        return None

    ibi = np.diff(wbeats)            # intervals in window
    t_iv = wbeats[1:]                # times at interval end
    idx = np.arange(ar_order, ibi.size)
    if idx.size == 0:
        return None

    Hcols = [ibi[idx - j] for j in range(1, ar_order + 1)]  # w_{i-1}..w_{i-p}
    H = np.column_stack(Hcols)
    if has_theta0:
        H = np.column_stack([np.ones(idx.size), H])         # theta0 + AR terms

    x = ibi[idx]                                            # targets
    t = t_iv[idx]
    wloc = np.exp(-alpha * (t_right - t))                   # Eq. 8 weights

    # survivor term info for last incomplete interval (Eq. 8)
    r_resid = max(0.0, float(t_right - wbeats[-1]))
    H_surv = None
    if r_resid > 0 and ibi.size >= ar_order:
        last_hist = ibi[-ar_order:][::-1]                   # w_{n-1}..w_{n-p}
        if has_theta0:
            H_surv = np.r_[1.0, last_hist]
        else:
            H_surv = last_hist
    return H, x, wloc, r_resid, H_surv

def _ig_logpdf_terms(x, mu, lam, eps=1e-12):
    # Returns per-sample logpdf using Barbieri params (mean=mu, var=mu^3/lam)
    return 0.5*(np.log(lam) - np.log(2*np.pi) - 3*np.log(x + eps)) \
         - lam * ((x - mu)**2) / (2*(mu**2) * (x + eps))

def _neg_local_ll(theta, H, x, wloc, r_resid, H_surv, alpha,
                  mu_floor=0.2, lam_floor=1e-6, pen=1e6, eps=1e-12,
                  use_huber=False, huber_c=1.5, ridge=0.0):
    """
    Numerically safe negative local log-likelihood (Barbieri IG).
    - Soft barrier if mu <= mu_floor
    - Clamp lambda to lam_floor
    - Survivor term weighted by exp(-alpha * r_resid)
    - Supports robust weights and ridge penalty
    """
    lam = max(float(theta[-1]), lam_floor)
    beta = theta[:-1]

    mu = H @ beta
    # quadratic penalty if any mu is too small
    bad = mu <= mu_floor
    mu_safe = np.where(bad, mu_floor, mu)

    # Robust weighting (Huber) on residuals
    if use_huber:
        r = (x - mu_safe) / np.maximum(mu_safe, 1e-6)
        c = float(huber_c)
        w_rob = np.where(np.abs(r) <= c, 1.0, c/np.maximum(np.abs(r), 1e-12))
        w_eff = wloc * w_rob
    else:
        w_eff = wloc

    # core weighted log-likelihood
    term = 0.5*(np.log(lam) - np.log(2*np.pi) - 3*np.log(x + eps)) \
         - lam * ((x - mu_safe)**2) / (2 * (mu_safe**2) * (x + eps))
    ll = np.sum(w_eff * term)

    if np.any(bad):
        ll -= pen * float(np.sum((mu[bad] - mu_floor)**2))

    # Ridge penalty on AR coefficients (not on intercept)
    if ridge > 0.0 and H.shape[1] >= 2:
        ll -= float(ridge) * float(np.sum(beta[1:]**2))

    # survivor term (Eq. 8) for last incomplete interval
    if H_surv is not None and r_resid > 0.0:
        mu_last = float(H_surv @ beta)
        mu_last = max(mu_last, mu_floor)
        S = 1.0 - invgauss.cdf(r_resid, mu=mu_last/lam, scale=lam)
        S = np.clip(S, 1e-12, 1.0)
        ll += np.exp(-alpha * r_resid) * np.log(S)

    return -float(ll)

def _neg_local_ll_and_grad(theta, H, x, wloc, r_resid, H_surv, alpha,
                           mu_floor=1e-8, lam_floor=1e-8, pen=1e6, eps=1e-12,
                           use_huber=False, huber_c=1.5, ridge=0.0):
    """
    Analytic gradient version.
    theta = [beta..., lam]
    Returns (negLL, grad) where grad matches Barbieri IG derivatives:
      dℓ/dlam  = 0.5 * Σ wloc * ( -1/lam + (x-mu)^2 / (mu^2 * x) )
      dℓ/dbeta = X^T [ -lam * wloc * ( (x-mu) / mu^3 ) ]   (with mu = X beta)
    Survivor term uses weight exp(-alpha*r_resid) and finite-diff gradient.
    Supports robust weights and ridge penalty.
    """
    lam = max(float(theta[-1]), lam_floor)
    beta = theta[:-1]

    mu = H @ beta
    bad = mu <= mu_floor
    mu_safe = np.where(bad, mu_floor, mu)

    if use_huber:
        r = (x - mu_safe) / np.maximum(mu_safe, 1e-6)
        c = float(huber_c)
        w_rob = np.where(np.abs(r) <= c, 1.0, c/np.maximum(np.abs(r), 1e-12))
        w_eff = wloc * w_rob
    else:
        w_eff = wloc

    # core weighted log-likelihood
    term = 0.5*(np.log(lam) - np.log(2*np.pi) - 3*np.log(x + eps)) \
         - lam * ((x - mu_safe)**2) / (2 * (mu_safe**2) * (x + eps))
    ll = np.sum(w_eff * term)

    if np.any(bad):
        ll -= pen * float(np.sum((mu[bad] - mu_floor)**2))

    # ---- analytic gradient (core) ----
    # dℓ/dlam
    dL_dlam = 0.5 * np.sum(w_eff * (-1.0/lam + ((x - mu_safe)**2) / ((mu_safe**2) * (x + eps))))
    # dℓ/dbeta
    resid_mu3 = (x - mu_safe) / (np.maximum(mu_safe, mu_floor)**3)
    dL_dbeta = H.T @ (-lam * w_eff * resid_mu3)

    ll_total = ll
    g_beta = dL_dbeta.copy()
    g_lam = dL_dlam

    # Ridge penalty on AR coefficients (not on intercept)
    if ridge > 0.0 and H.shape[1] >= 2:
        ll_total -= float(ridge) * float(np.sum(beta[1:]**2))
        g_beta[1:] += 2.0 * float(ridge) * beta[1:]

    # ---- survivor term (Eq. 8) ----
    if H_surv is not None and r_resid > 0.0:
        mu_last = float(H_surv @ beta)
        mu_last = max(mu_last, mu_floor)
        S = 1.0 - invgauss.cdf(r_resid, mu=mu_last/lam, scale=lam)
        S = np.clip(S, 1e-12, 1.0)
        w_res = np.exp(-alpha * r_resid)
        ll_total += w_res * np.log(S)

        # finite-diff gradient of log S wrt mu_last and lam
        def logS(mul, lamv):
            return np.log(np.clip(1.0 - invgauss.cdf(r_resid, mu=mul/lamv, scale=lamv), 1e-12, 1.0))
        dmu = max(1e-6, 1e-4 * mu_last)
        dlam = max(1e-6, 1e-4 * lam)
        dlogS_dmu = (logS(mu_last + dmu, lam) - logS(mu_last - dmu, lam)) / (2*dmu)
        dlogS_dlam = (logS(mu_last, lam + dlam) - logS(mu_last, lam - dlam)) / (2*dlam)
        g_beta += w_res * dlogS_dmu * H_surv
        g_lam  += w_res * dlogS_dlam

    negLL = -float(ll_total)
    grad  = -np.r_[g_beta, g_lam]
    return negLL, grad

# ---- Analytic Hessian (core IG terms) + Hessian-vector product ----
def _neg_local_hessian(theta, H, x, wloc, mu_floor=1e-8, lam_floor=1e-8, eps=1e-12,
                       use_huber=False, huber_c=1.5, ridge=0.0):
    """Full Hessian of the negative weighted local log-likelihood (core IG terms).
    C++ uses [kappa, theta...] for ℓ; here we use [beta..., lam] for f=-ℓ.
    Blocks for f=-ℓ:
      H_{beta,beta} = - X^T diag( lam * wloc * ( (3x - 2mu) / mu^4 ) ) X
      H_{beta,lam}  =   X^T( wloc * (x - mu) / mu^3 )
      H_{lam,lam}   = - 0.5 * sum(wloc) / lam^2
    Survivor term is omitted here for stability (small contribution); gradient still includes it.
    Supports robust weights and ridge penalty.
    """
    lam = max(float(theta[-1]), lam_floor)
    beta = theta[:-1]

    mu = H @ beta
    mu_safe = np.where(mu <= mu_floor, mu_floor, mu)

    if use_huber:
        r = (x - mu_safe) / np.maximum(mu_safe, 1e-6)
        c = float(huber_c)
        w_rob = np.where(np.abs(r) <= c, 1.0, c/np.maximum(np.abs(r), 1e-12))
        w_eff = wloc * w_rob
    else:
        w_eff = wloc

    # Diagonal weights per sample
    a = lam * w_eff * ((3.0 * x - 2.0 * mu_safe) / (mu_safe**4))   # for beta-beta block (with sign -)
    d = w_eff * ((x - mu_safe) / (mu_safe**3))                      # for beta-lam block

    # Assemble blocks
    Hbb = -(H.T * a) @ H                     # (p+1)x(p+1) or pxp depending on has_theta0
    Hbl = H.T @ d                            # (p+1)x1
    Hll = -0.5 * float(np.sum(w_eff)) / (lam**2)  # scalar

    # Ridge penalty on AR coefficients (not on intercept)
    if ridge > 0.0 and H.shape[1] >= 2:
        Hbb[1:, 1:] += 2.0 * float(ridge) * np.eye(H.shape[1]-1)

    n_beta = H.shape[1]
    H_full = np.zeros((n_beta + 1, n_beta + 1), dtype=float)
    H_full[:n_beta, :n_beta] = Hbb
    H_full[:n_beta, -1] = Hbl
    H_full[-1, :n_beta] = Hbl.T
    H_full[-1, -1] = Hll
    return H_full


def _neg_local_hessp(theta, v, H, x, wloc, mu_floor=1e-8, lam_floor=1e-8, eps=1e-12,
                     use_huber=False, huber_c=1.5, ridge=0.0):
    """Hessian–vector product for Newton-CG: returns (∇²f)·v."""
    return _neg_local_hessian(theta, H, x, wloc, mu_floor=mu_floor, lam_floor=lam_floor, eps=eps,
                               use_huber=use_huber, huber_c=huber_c, ridge=ridge) @ v

# ------- API: single regression -------
def compute_single_regression(
    events, ar_order=9, has_theta0=True, right_censoring=False,
    alpha=0.02, distribution=Distributions.InverseGaussian, max_iter=200,
    use_wls_init=False, use_huber=False, huber_c=1.5, ridge=0.0
) -> RegressionResult:
    assert distribution == Distributions.InverseGaussian, "Only IG supported in this port."

    events = np.asarray(events, float)
    t_right = events[-1]                            # use the last beat as window edge
    window_length = events[-1] - events[0]         # use full segment
    design = _window_design(events, ar_order, t_right, window_length, alpha, has_theta0)
    if design is None:
        raise ValueError("Not enough data for this ar_order.")
    H, x, wloc, r_resid, H_surv = design

    # init params
    mu0 = float(np.clip(np.median(x), 0.3, 1.5))
    if use_wls_init:
        W = np.sqrt(wloc)[:, None]
        Hw = H * W
        xw = x * W.ravel()
        try:
            beta0, *_ = np.linalg.lstsq(Hw, xw, rcond=None)
        except np.linalg.LinAlgError:
            beta0 = np.zeros(H.shape[1]);
            if has_theta0: beta0[0] = mu0
    else:
        if has_theta0:
            beta0 = np.zeros(H.shape[1]); beta0[0] = mu0
        else:
            beta0 = np.zeros(H.shape[1])

    lam0 = 5.0
    theta0 = np.r_[beta0, lam0]

    # objective (include survivor term if requested)
    def obj(theta):
        return _neg_local_ll(theta, H, x, wloc,
                         r_resid if right_censoring else 0.0,
                         H_surv if right_censoring else None,
                         alpha=alpha, use_huber=use_huber, huber_c=huber_c, ridge=ridge)

    # Try Newton-CG with analytic gradient (no bounds). Fallback to L-BFGS-B with bounds.
    def fun(theta):
        f, _ = _neg_local_ll_and_grad(theta, H, x, wloc,
                                      r_resid if right_censoring else 0.0,
                                      H_surv if right_censoring else None,
                                      alpha=alpha,
                                      use_huber=use_huber, huber_c=huber_c, ridge=ridge)
        return f

    def jac(theta):
        _, g = _neg_local_ll_and_grad(theta, H, x, wloc,
                                      r_resid if right_censoring else 0.0,
                                      H_surv if right_censoring else None,
                                      alpha=alpha,
                                      use_huber=use_huber, huber_c=huber_c, ridge=ridge)
        return g

    try:
        res = minimize(
            fun, theta0,
            method="Newton-CG",
            jac=jac,
            hessp=lambda th, vv: _neg_local_hessp(th, vv, H, x, wloc,
                                                  use_huber=use_huber, huber_c=huber_c, ridge=ridge),
            options=dict(maxiter=max_iter, xtol=1e-6, disp=False),
        )
        if not res.success:
            raise RuntimeError("Newton-CG did not converge")
    except Exception:
        n_beta = H.shape[1]
        bnds = [(0.3, 1.8)] + [(None, None)]*(n_beta-1) + [(1e-4, 1e3)]
        res = minimize(obj, theta0, method="L-BFGS-B", bounds=bnds,
                       options=dict(maxiter=max_iter))
        if not res.success:
            raise RuntimeError("Optimization failed: " + res.message)

    theta_hat = res.x
    lam = theta_hat[-1]
    beta = theta_hat[:-1]

    # instantaneous estimates at right edge
    # Build H_t from last p intervals
    mask = (events > t_right - window_length) & (events <= t_right)
    ibi = np.diff(events[mask])
    if has_theta0:
        H_t = np.r_[1.0, ibi[-ar_order:][::-1]]
    else:
        H_t = ibi[-ar_order:][::-1]
    mu_t = float(H_t @ beta)
    mu_t = max(mu_t, 1e-6)
    sd_t = float(np.sqrt(mu_t**3 / lam))

    return RegressionResult(
        thetap=beta[1:] if has_theta0 else beta,
        theta0=beta[0] if has_theta0 else 0.0,
        kappa=lam,
        likelihood=-res.fun,
        mean_interval=mu_t,
        sigma=sd_t
    )

# ------- API: full regression -------
def compute_full_regression(
    events, window_length=60.0, delta=0.005, ar_order=9,
    has_theta0=True, right_censoring=True, alpha=0.02,
    distribution=Distributions.InverseGaussian, max_iter=200,
    use_wls_init=False, use_huber=False, huber_c=1.5, ridge=0.0
) -> Result:
    assert distribution == Distributions.InverseGaussian, "Only IG supported in this port."
    events = np.asarray(events, float)
    t0 = events[0] + window_length
    t1 = events[-1]

    times = []
    mus = []
    lam_series = []
    sds = []
    thetas = []
    taus_all = []  # placeholder: you can fill with time-rescaled residuals later

    # warm start
    theta_last = None

    for t_right in np.arange(t0, t1, delta):
        design = _window_design(events, ar_order, t_right, window_length, alpha, has_theta0)
        if design is None:
            continue
        H, x, wloc, r_resid, H_surv = design

        # init
        mu0 = float(np.clip(np.median(x), 0.3, 1.5))
        if use_wls_init:
            W = np.sqrt(wloc)[:, None]
            Hw = H * W
            xw = x * W.ravel()
            try:
                beta0, *_ = np.linalg.lstsq(Hw, xw, rcond=None)
            except np.linalg.LinAlgError:
                beta0 = np.zeros(H.shape[1]);
                if has_theta0: beta0[0] = mu0
        else:
            if has_theta0:
                beta0 = np.zeros(H.shape[1]); beta0[0] = mu0
            else:
                beta0 = np.zeros(H.shape[1])
        lam0 = 5.0
        theta0 = np.r_[beta0, lam0] if theta_last is None else theta_last

        # objective (include survivor term if requested)
        def obj(theta):
            return _neg_local_ll(theta, H, x, wloc,
                         r_resid if right_censoring else 0.0,
                         H_surv if right_censoring else None,
                         alpha=alpha, use_huber=use_huber, huber_c=huber_c, ridge=ridge)

        def fun(theta):
            f, _ = _neg_local_ll_and_grad(theta, H, x, wloc,
                                          r_resid if right_censoring else 0.0,
                                          H_surv if right_censoring else None,
                                          alpha=alpha,
                                          use_huber=use_huber, huber_c=huber_c, ridge=ridge)
            return f

        def jac(theta):
            _, g = _neg_local_ll_and_grad(theta, H, x, wloc,
                                          r_resid if right_censoring else 0.0,
                                          H_surv if right_censoring else None,
                                          alpha=alpha,
                                          use_huber=use_huber, huber_c=huber_c, ridge=ridge)
            return g

        # try:
        #     res = minimize(
        #         fun, theta0,
        #         method="Newton-CG",
        #         jac=jac,
        #         hessp=lambda th, vv: _neg_local_hessp(th, vv, H, x, wloc,
        #                                               use_huber=use_huber, huber_c=huber_c, ridge=ridge),
        #         options=dict(maxiter=max_iter, xtol=1e-6, disp=False),
        #     )
        #     if not res.success:
        #         raise RuntimeError("Newton-CG did not converge")
        # except Exception:
        n_beta = H.shape[1]
        bnds = [(0.3, 1.8)] + [(None, None)]*(n_beta-1) + [(1e-4, 1e3)]
        res = minimize(obj, theta0, method="L-BFGS-B", bounds=bnds,
                        options=dict(maxiter=max_iter))
        if not res.success:
            continue

        theta_last = res.x
        lam = theta_last[-1]
        beta = theta_last[:-1]

        # instantaneous (right-edge)
        ibi = np.diff(events[(events > t_right - window_length) & (events <= t_right)])
        if has_theta0:
            H_t = np.r_[1.0, ibi[-ar_order:][::-1]]
        else:
            H_t = ibi[-ar_order:][::-1]
        mu_t = float(H_t @ beta)
        mu_t = max(mu_t, 1e-6)
        sd_t = float(np.sqrt(mu_t**3 / lam))

        times.append(t_right)
        mus.append(mu_t)
        lam_series.append(float(lam))
        sds.append(sd_t)
        thetas.append((beta[0] if has_theta0 else 0.0, beta[1:] if has_theta0 else beta))

    return Result(Time=np.array(times), Mu=np.array(mus),
                  lambda_=np.array(lam_series), sd_RR=np.array(sds),
                  thetas=thetas, taus=np.array(taus_all))