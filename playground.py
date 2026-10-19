"""
happy-control playground  --  MATLAB-style interactive workflow
================================================================

Start it from the repo root with:

    python -i playground.py

Everything below is then loaded in memory and you can poke at it, e.g.

    >>> ls()                  # list all example systems
    >>> s = get('duffing')    # fresh ControlSystem object
    >>> show(s)               # pretty-print the system
    >>> s = approx('sine')    # compute the approximating system
    >>> plot('sine')          # plot trajectories comparing real vs approx
    >>> plot('kapitza')       # phase space + coordinate evolution + error
    >>> plot('scalar')        # 1D trajectory comparison
    >>> s.b_1                 # inspect any attribute afterwards
    >>> s.main_part_lie       # Lie elements of the principal part
    >>> s.ideal               # generating set of the ideal
    >>> pdf(s)                # -> pdfs/sine.pdf
    >>> run('kapitza')        # approx + pdf in one call
    >>> run_all()             # every system -> pdfs/

Define your own system on the fly:

    >>> a = Matrix([x2, -sin(t)*x1**2])
    >>> b = Matrix([1, 0])
    >>> s = approx(ControlSystem(a, b), name='my_test')
    >>> plot(s)
    >>> pdf(s)

NOTES
  * The algorithm expands around x = 0, t = 0 (nonlinear power moments),
    so systems should satisfy a(0, 0) = 0 and b(0, 0) != 0.
  * State variables MUST be the symbols x_{1}, x_{2}, ... (exported here as
    x1, x2, ...) and time MUST be `t`.
  * If a system needs very high-order Lie brackets (or is not "controllable"
    in this sense), the computation can take forever -> Ctrl+C and rethink.
  * Fliess mode (fliess=True) treats the system as driftless with two
    inputs:  x' = a(t,x) u1 + b(t,x) u2, expanded around `init_point`.
  * Each ControlSystem accumulates results, so don't call approx() twice on
    the same object -- use get(name) / a new ControlSystem(...) instead.
"""

import os
import subprocess
import time
from contextlib import contextmanager
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import sympy as sym
from sympy import Matrix, Rational, sin, cos, exp, sqrt, atan, tanh  # noqa: F401  (handy in the REPL)

from happy_control import approximation_tools as _at
from happy_control.approximation_tools import ControlSystem

# Enable interactive plotting mode so plot(...) does not block python -i
try:
    plt.ion()
except Exception:
    pass

# --------------------------------------------------------------------------
# Symbols  (names must match what ControlSystem.get_variables() generates)
# --------------------------------------------------------------------------
x1, x2, x3, x4 = sym.symbols('x_{1} x_{2} x_{3} x_{4}')
t = ControlSystem.t
u, u1, u2 = sym.symbols('u u_1 u_2')

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(REPO_DIR, 'pdfs')


# --------------------------------------------------------------------------
# Example systems
#   power-moments mode:  x' = a(t, x) + b(t, x) u
#   Fliess mode:         x' = a(t, x) u1 + b(t, x) u2
# Each factory returns (a, b, ControlSystem kwargs). Factories instead of
# objects because ControlSystem accumulates state.
# --------------------------------------------------------------------------
def _scalar():
    # 1D nonlinear drift: x1' = -x1 + sin(t) * x1^2 + u
    a = Matrix([-x1 + sin(t) * x1**2])
    b = Matrix([1])
    return a, b, {}


def _kapitza():
    # Pendulum on a vertically vibrating pivot (Kapitza pendulum), light
    # damping, torque input. Linearisation is controllable -> baseline case:
    # the approximation collapses to the linear chain  x1' = -u, x2' = -t u.
    a = Matrix([x2,
                -(1 + 2 * cos(t)) * sin(x1) - Rational(1, 10) * x2])
    b = Matrix([0, 1])
    return a, b, {}


def _sine():
    # Linearisation is NOT controllable: x2 only "feels" x1 quadratically,
    # and only through a time-varying gain. Forces a genuinely nonlinear
    # homogeneous approximation (principal part reaches order 4).
    a = Matrix([0,
                sin(t) * x1**2 + x1**3])
    b = Matrix([1, 0])
    return a, b, {}


def _duffing():
    # Duffing oscillator with Mathieu-pumped stiffness, driven by an
    # electromagnet: force ~ current^2, current x3 follows RL dynamics.
    # Nonlinear approximation AND a stationary approximation exists.
    a = Matrix([x2,
                -(1 + sin(t)) * x1 - x1**3 + x3**2,
                -x3])
    b = Matrix([0, 0, 1])
    return a, b, {}


def _spin():
    # Toy rigid-body-like rotation coupling with time-varying cross terms
    # and a state-dependent input direction (b depends on x).
    a = Matrix([0,
                t * x1 + x1 * x3,
                -x1 * x2 + t**2 * x1])
    b = Matrix([1, x3, -x2])
    return a, b, {}


def _cascade():
    # Three-state cascade: exponentially growing coupling into x2,
    # bilinear / quadratic coupling into x3. Principal part up to order 5.
    a = Matrix([0,
                (exp(t) - 1) * x1,
                x1 * x2 + cos(t) * x1**2 - x1**2])
    b = Matrix([1, 0, 0])
    return a, b, {}


def _chain():
    # Four-state chain mixing time-varying linear, quadratic and bilinear
    # couplings.
    a = Matrix([0,
                x1,
                sin(t) * x1**2,
                x2 * x3 + t * x2])
    b = Matrix([1, 0, 0, 0])
    return a, b, {}


def _unicycle():
    # FLIESS MODE (driftless, two inputs): unicycle whose drive gain grows
    # in time.  u1 = forward drive, u2 = steering rate.
    # Approximation is the Brockett nonholonomic integrator.
    a = Matrix([(1 + t) * cos(x3),
                (1 + t) * sin(x3),
                0])
    b = Matrix([0, 0, 1])
    return a, b, {}


# Simple, memorable system names:
# name -> (description, factory, fliess)
SYSTEMS = {
    'scalar':   ('1D nonlinear drift -x1 + sin(t)*x1^2, control u', _scalar, False),
    'kapitza':  ('2D Kapitza pendulum, baseline (linearly controllable)', _kapitza, False),
    'sine':     ('2D linearly uncontrollable, x2 ~ sin(t) x1^2', _sine, False),
    'duffing':  ('3D Mathieu-Duffing driven by force ~ current^2', _duffing, False),
    'spin':     ('3D rotation-like coupling, state-dependent b', _spin, False),
    'cascade':  ('3D exp(t) coupling + bilinear cascade', _cascade, False),
    'chain':    ('4D mixed linear/quadratic/bilinear chain', _chain, False),
    'unicycle': ('3D unicycle, time-varying drive  [Fliess mode]', _unicycle, True),
}

# Fast querying aliases for convenience in interactive mode
ALIASES = {
    'electromagnet_duffing': 'duffing',
    'duff':                  'duffing',
    'sine_quadratic':        'sine',
    'sine_quad':             'sine',
    'pendulum':              'kapitza',
    'spinning_body':         'spin',
    'spinning':              'spin',
    'exp_cascade':           'cascade',
    'quad_chain_4d':         'chain',
    'quad_chain':            'chain',
    'quad':                  'chain',
    'unicycle_fliess':       'unicycle',
    'riccati':               'scalar',
    '1d':                    'scalar',
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ls():
    """List the available example systems."""
    w = max(map(len, SYSTEMS))
    for name, (desc, _, _) in SYSTEMS.items():
        print(f'  {name:<{w}}  {desc}')


def get(name):
    """Return a FRESH ControlSystem for the named example."""
    clean = str(name).strip().lower()
    clean = ALIASES.get(clean, clean)
    if clean not in SYSTEMS:
        matches = [k for k in SYSTEMS if k.startswith(clean)]
        if len(matches) == 1:
            clean = matches[0]
        else:
            raise KeyError(f'Unknown system {name!r}. Call ls() to see options: {list(SYSTEMS.keys())}')
    _, factory, fliess = SYSTEMS[clean]
    a, b, kw = factory()
    s = ControlSystem(a, b, **kw)
    s.name = clean
    s._fliess = fliess
    return s


def _resolve(sys_or_name):
    return get(sys_or_name) if isinstance(sys_or_name, str) else sys_or_name


def _pp(expr):
    try:
        sym.pprint(expr, use_unicode=True)
    except UnicodeEncodeError:
        sym.pprint(expr, use_unicode=False)


def _rhs(first, second, fliess):
    return first * u1 + second * u2 if fliess else first + second * u


def show(sys_or_name, fliess=None):
    """Pretty-print the system (drift + control, or Fliess two-input form)."""
    s = _resolve(sys_or_name)
    fliess = getattr(s, '_fliess', False) if fliess is None else fliess
    form = "a(t,x) u1 + b(t,x) u2" if fliess else "a(t,x) + b(t,x) u"
    print(f"\n{getattr(s, 'name', 'system')}:  x' = {form}\n")
    _pp(sym.Eq(Matrix(sym.symbols(f"x'_1:{s.dim + 1}")), _rhs(s.a, s.b, fliess)))
    print()
    return s


def approx(sys_or_name, fliess=None, name=None, verbose=True):
    """
    Compute the approximating system. Accepts a name from SYSTEMS or a
    ControlSystem instance. Returns the (now populated) ControlSystem.
    fliess=None -> use the system's default (False for custom systems).
    """
    s = _resolve(sys_or_name)
    if name:
        s.name = name
    fliess = getattr(s, '_fliess', False) if fliess is None else fliess
    s._fliess = fliess
    if s.main_part_lie:
        raise RuntimeError('This ControlSystem was already approximated '
                           '(results accumulate). Create a fresh one.')
    if verbose:
        show(s)
    t0 = time.perf_counter()
    s.calc_approx_system(fliess=fliess)
    dt = time.perf_counter() - t0
    if verbose:
        print(f'--- approximation done in {dt:.2f}s  '
              f'(principal part orders: {list(s.main_part_lie)})\n')
        print('Approximating system  ({}):'.format(
            'b_0 u1 + b_1 u2' if fliess else 'b_0 + b_1 u'))
        _pp(_rhs(s.b_0, s.b_1, fliess))
        if not fliess:
            print('\nStationary approximating system:')
            if s.b_0_stationary is None or s.b_1_stationary is None:
                print('  does not exist')
            else:
                _pp(s.b_0_stationary + s.b_1_stationary * u)
        print()
    return s


# --------------------------------------------------------------------------
# Simulation & Trajectory Plotting
# --------------------------------------------------------------------------
def _unify_expr(expr, dim, state_syms):
    """Normalize any alternative symbol variants (e.g. x1, x_1) into canonical x_{i}."""
    if expr is None:
        return None
    subs = {}
    for s in expr.free_symbols:
        name = s.name
        for i in range(1, dim + 1):
            if name in (f'x_{i}', f'x{i}', f'x_{{{i}}}'):
                subs[s] = state_syms[i - 1]
                break
    return sym.sympify(expr).subs(subs)


def plot(
    sys_or_name,
    t_span=(0.0, 2.5),
    x0=None,
    u=None,
    num_pts=300,
    save_path=None,
    show_plot=True,
    figsize=None,
    seed=None,
):
    """
    Simulate the differential equations of the real system and its approximation(s)
    using SciPy's high-precision RK45 solver and plot their trajectories without grids.

    Parameters
    ----------
    sys_or_name : ControlSystem or str
        System instance or name from SYSTEMS (e.g. 'sine', 'kapitza', 'duffing', 'scalar').
        If not yet approximated, approx(...) is computed automatically.
    t_span : tuple of (float, float)
        Time integration interval (t_start, t_end). Defaults to (0.0, 2.5).
    x0 : array-like, optional
        Initial state vector. If None, a non-zero initial condition is randomly sampled
        near the origin (the approximation expansion point).
    u : float, callable, or tuple, optional
        Control input function u(t) or constant value.
        For Fliess mode, a callable returning (u1, u2) or a tuple of two inputs.
        Defaults to cos(2*t) for single input, or (cos(t), sin(t)) for Fliess mode.
    num_pts : int
        Number of evaluation points along the trajectory. Defaults to 300.
    save_path : str, optional
        Path where the generated figure will be saved (e.g. 'trajectory.png').
    show_plot : bool
        Whether to display the plot window (non-blocking in interactive mode).
    figsize : tuple, optional
        Figure size in inches (width, height).
    seed : int, optional
        Random seed for reproducible initial condition generation.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    s = _resolve(sys_or_name)
    if not getattr(s, 'main_part_lie', None):
        approx(s, verbose=False)

    dim = s.dim
    fliess = getattr(s, '_fliess', False)
    name = getattr(s, 'name', 'system')

    # Check whether stationary approximation exists
    has_stat = (
        not fliess
        and getattr(s, 'b_0_stationary', None) is not None
        and getattr(s, 'b_1_stationary', None) is not None
        and not (s.b_0_stationary == sym.zeros(dim, 1) and s.b_1_stationary == sym.zeros(dim, 1))
    )

    # Sample random initial condition near origin if not supplied
    if seed is not None:
        np.random.seed(seed)
    if x0 is None:
        x0 = np.random.uniform(-0.35, 0.35, size=dim)
        for i in range(dim):
            if abs(x0[i]) < 0.05:
                x0[i] = 0.15 if x0[i] >= 0 else -0.15
    else:
        x0 = np.asarray(x0, dtype=float).flatten()
        if len(x0) != dim:
            raise ValueError(f"Initial condition x0 has dimension {len(x0)}, expected {dim}")

    # Configure control input u(t)
    if u is None:
        if fliess:
            u_func = lambda t: (float(np.cos(t)), float(np.sin(t)))
        else:
            u_func = lambda t: float(np.cos(2.0 * t))
    elif callable(u):
        u_func = u
    elif isinstance(u, (int, float)):
        val = float(u)
        u_func = lambda t: val
    elif isinstance(u, (list, tuple)) and len(u) == 2:
        u1_f = u[0] if callable(u[0]) else (lambda t, v=u[0]: float(v))
        u2_f = u[1] if callable(u[1]) else (lambda t, v=u[1]: float(v))
        u_func = lambda t: (u1_f(t), u2_f(t))
    else:
        u_func = lambda t: u

    # Setup symbols and lambdified functions
    t_sym = sym.Symbol('t')
    state_syms = [sym.Symbol(f'x_{i+1}') for i in range(dim)]
    u_sym = sym.Symbol('u')
    u1_sym, u2_sym = sym.symbols('u_1 u_2')

    if fliess:
        real_expr = [_unify_expr(s.a[i] * u1_sym + s.b[i] * u2_sym, dim, state_syms) for i in range(dim)]
        approx_expr = [_unify_expr(s.b_0[i] * u1_sym + s.b_1[i] * u2_sym, dim, state_syms) for i in range(dim)]
        f_real = sym.lambdify([t_sym, state_syms, u1_sym, u2_sym], real_expr, modules=['numpy', 'math'])
        f_approx = sym.lambdify([t_sym, state_syms, u1_sym, u2_sym], approx_expr, modules=['numpy', 'math'])

        def ode_real(t, x):
            u1_val, u2_val = u_func(t)
            return np.asarray(f_real(t, list(x), u1_val, u2_val), dtype=float).ravel()

        def ode_approx(t, x):
            u1_val, u2_val = u_func(t)
            return np.asarray(f_approx(t, list(x), u1_val, u2_val), dtype=float).ravel()
    else:
        real_expr = [_unify_expr(s.a[i] + s.b[i] * u_sym, dim, state_syms) for i in range(dim)]
        approx_expr = [_unify_expr(s.b_0[i] + s.b_1[i] * u_sym, dim, state_syms) for i in range(dim)]
        f_real = sym.lambdify([t_sym, state_syms, u_sym], real_expr, modules=['numpy', 'math'])
        f_approx = sym.lambdify([t_sym, state_syms, u_sym], approx_expr, modules=['numpy', 'math'])

        def ode_real(t, x):
            u_val = u_func(t)
            return np.asarray(f_real(t, list(x), u_val), dtype=float).ravel()

        def ode_approx(t, x):
            u_val = u_func(t)
            return np.asarray(f_approx(t, list(x), u_val), dtype=float).ravel()

        if has_stat:
            stat_expr = [_unify_expr(s.b_0_stationary[i] + s.b_1_stationary[i] * u_sym, dim, state_syms) for i in range(dim)]
            f_stat = sym.lambdify([t_sym, state_syms, u_sym], stat_expr, modules=['numpy', 'math'])

            def ode_stat(t, x):
                u_val = u_func(t)
                return np.asarray(f_stat(t, list(x), u_val), dtype=float).ravel()

    # Numerical integration with RK45
    t_eval = np.linspace(t_span[0], t_span[1], num_pts)
    sol_r = solve_ivp(ode_real, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-7, atol=1e-9)
    sol_a = solve_ivp(ode_approx, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-7, atol=1e-9)
    sol_s = solve_ivp(ode_stat, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-7, atol=1e-9) if has_stat else None

    # Sync time points across all simulations (handling early termination if any)
    t_end = min(sol_r.t[-1], sol_a.t[-1])
    if has_stat:
        t_end = min(t_end, sol_s.t[-1])

    idx_r = np.where(sol_r.t <= t_end + 1e-9)[0]
    idx_a = np.where(sol_a.t <= t_end + 1e-9)[0]
    n_pts = min(len(idx_r), len(idx_a))
    if has_stat:
        idx_s = np.where(sol_s.t <= t_end + 1e-9)[0]
        n_pts = min(n_pts, len(idx_s))

    t_plot = sol_r.t[:n_pts]
    y_r = sol_r.y[:, :n_pts]
    y_a = sol_a.y[:, :n_pts]
    y_s = sol_s.y[:, :n_pts] if has_stat else None

    # Error / Divergence calculation
    err_a = np.linalg.norm(y_r - y_a, axis=0)
    err_s = np.linalg.norm(y_r - y_s, axis=0) if has_stat else None
    max_err_idx = np.argmax(err_a)
    max_err_val = err_a[max_err_idx]
    max_err_t = t_plot[max_err_idx]

    # Console summary
    x0_str = '[' + ', '.join(f'{v:+.3f}' for v in x0) + ']'
    print(f"\n[plot] System: {name} (dim={dim}) | x0 = {x0_str} | t in [{t_plot[0]:.2f}, {t_plot[-1]:.2f}]")
    print(f"       Max divergence ||x_real - x_approx||: {max_err_val:.4f} at t = {max_err_t:.2f}s")
    if has_stat:
        max_s_idx = np.argmax(err_s)
        print(f"       Max stationary divergence:          {err_s[max_s_idx]:.4f} at t = {t_plot[max_s_idx]:.2f}s")

    # Styling palette
    c_real = '#1f77b4'       # Blue
    c_approx = '#e11d48'     # Crimson
    c_stat = '#16a34a'       # Green
    c_start = '#10b981'      # Bright start marker
    colors_state = ['#1f77b4', '#d97706', '#9333ea', '#0891b2']

    if dim == 1:
        # 1D System: single trajectory comparison plot
        fig_sz = figsize or (7.2, 4.5)
        fig, ax1 = plt.subplots(figsize=fig_sz)

        # Trajectory
        ax1.plot(t_plot, y_r[0], label='Real system', color=c_real, lw=2.4)
        ax1.plot(t_plot, y_a[0], label='Homogeneous approx', color=c_approx, ls='--', lw=2.2)
        if has_stat:
            ax1.plot(t_plot, y_s[0], label='Stationary approx', color=c_stat, ls=':', lw=2.0)
        ax1.fill_between(t_plot, y_r[0], y_a[0], color=c_approx, alpha=0.12, label='Divergence region')
        ax1.plot(t_plot[0], y_r[0, 0], 'o', color=c_start, ms=8, zorder=5, label=f'Start x0 = {x0[0]:+.2f}')
        ax1.set_xlabel('Time t', fontsize=10)
        ax1.set_ylabel('State x1', fontsize=10)
        ax1.set_title(f'{name} (1D): Trajectory Comparison', fontsize=11, fontweight='bold')
        ax1.legend(frameon=True, fontsize=8)
        ax1.grid(False)

    elif dim == 2:
        # 2D System: 2-panel layout (Phase Plane, State Evolution over Time)
        fig_sz = figsize or (10.5, 4.8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_sz)

        # Panel 1: Phase Plane
        ax1.plot(y_r[0], y_r[1], label='Real system', color=c_real, lw=2.4)
        ax1.plot(y_a[0], y_a[1], label='Homogeneous approx', color=c_approx, ls='--', lw=2.2)
        if has_stat:
            ax1.plot(y_s[0], y_s[1], label='Stationary approx', color=c_stat, ls=':', lw=2.0)
        ax1.plot([x0[0]], [x0[1]], 'o', color=c_start, ms=8, zorder=5, label=f'Start x0=({x0[0]:+.2f}, {x0[1]:+.2f})')
        ax1.plot([y_r[0, -1]], [y_r[1, -1]], 's', color=c_real, ms=5, label='Real end')
        ax1.plot([y_a[0, -1]], [y_a[1, -1]], '^', color=c_approx, ms=5, label='Approx end')

        # Dotted connector lines indicating state-space divergence vectors
        step = max(1, len(t_plot) // 8)
        for idx in range(step, len(t_plot), step):
            ax1.plot([y_r[0, idx], y_a[0, idx]], [y_r[1, idx], y_a[1, idx]], color='#9ca3af', ls=':', lw=1.0)

        ax1.set_xlabel('State x1', fontsize=10)
        ax1.set_ylabel('State x2', fontsize=10)
        ax1.set_title(f'{name} (2D): Phase Space Trajectory', fontsize=11, fontweight='bold')
        ax1.legend(frameon=True, fontsize=8)
        ax1.grid(False)

        # Panel 2: States vs Time
        ax2.plot(t_plot, y_r[0], label='x1 (Real)', color=c_real, lw=2.0)
        ax2.plot(t_plot, y_a[0], label='x1 (Approx)', color=c_real, ls='--', lw=1.8)
        ax2.plot(t_plot, y_r[1], label='x2 (Real)', color=colors_state[1], lw=2.0)
        ax2.plot(t_plot, y_a[1], label='x2 (Approx)', color=colors_state[1], ls='--', lw=1.8)
        if has_stat:
            ax2.plot(t_plot, y_s[0], label='x1 (Stat)', color=c_stat, ls=':', lw=1.6)
            ax2.plot(t_plot, y_s[1], label='x2 (Stat)', color=colors_state[2], ls=':', lw=1.6)
        ax2.set_xlabel('Time t', fontsize=10)
        ax2.set_ylabel('State coordinates', fontsize=10)
        ax2.set_title('Coordinate Evolution over Time', fontsize=11, fontweight='bold')
        ax2.legend(frameon=True, fontsize=8)
        ax2.grid(False)

    else:
        # Higher-dimensional systems: 2-panel layout
        fig_sz = figsize or (10.5, 4.8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_sz)

        # Panel 1: (x1, x2) Projection
        ax1.plot(y_r[0], y_r[1], label='Real (x1, x2)', color=c_real, lw=2.4)
        ax1.plot(y_a[0], y_a[1], label='Approx (x1, x2)', color=c_approx, ls='--', lw=2.2)
        ax1.plot([x0[0]], [x0[1]], 'o', color=c_start, ms=8, zorder=5, label='Start (x1, x2)')
        ax1.set_xlabel('State x1', fontsize=10)
        ax1.set_ylabel('State x2', fontsize=10)
        ax1.set_title(f'{name} ({dim}D): x1-x2 Projection', fontsize=11, fontweight='bold')
        ax1.legend(frameon=True, fontsize=8)
        ax1.grid(False)

        # Panel 2: States vs Time
        for i in range(dim):
            c_st = colors_state[i % len(colors_state)]
            ax2.plot(t_plot, y_r[i], color=c_st, lw=1.8, label=f'x{i+1} Real')
            ax2.plot(t_plot, y_a[i], color=c_st, ls='--', lw=1.6, label=f'x{i+1} Approx')
        ax2.set_xlabel('Time t', fontsize=10)
        ax2.set_ylabel('State coordinates', fontsize=10)
        ax2.set_title('All States vs Time', fontsize=11, fontweight='bold')
        ax2.legend(frameon=True, fontsize=7)
        ax2.grid(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[plot] Figure saved: {save_path}")
    if show_plot:
        plt.show()

    return fig


# --------------------------------------------------------------------------
# PDF generation & batch execution
# --------------------------------------------------------------------------
@contextmanager
def _quiet_popen():
    """Silence pdflatex's console spam (generate_pdf uses subprocess.Popen)."""
    original = subprocess.Popen
    _at.subprocess.Popen = partial(original, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
    try:
        yield
    finally:
        _at.subprocess.Popen = original


def pdf(s, filename=None, quiet=True):
    """Generate a PDF report for an already-approximated system into pdfs/."""
    if not s.main_part_lie:
        raise RuntimeError('Run approx(...) on this system first.')
    os.makedirs(PDF_DIR, exist_ok=True)
    filename = filename or getattr(s, 'name', 'output')
    fliess = getattr(s, '_fliess', False)
    cwd = os.getcwd()                       # generate_pdf() chdir's around
    t0 = time.perf_counter()
    try:
        if quiet:
            with _quiet_popen():
                s.generate_pdf(path=PDF_DIR, filename=filename, fliess=fliess)
        else:
            s.generate_pdf(path=PDF_DIR, filename=filename, fliess=fliess)
    except Exception as e:
        print(f'PDF generation failed: {e!r}\n'
              f'Check {os.path.join(PDF_DIR, filename + ".log")} '
              f'or retry with pdf(s, quiet=False).')
        raise
    finally:
        os.chdir(cwd)
    out = os.path.join(PDF_DIR, filename + '.pdf')
    print(f'PDF written: {out}  ({time.perf_counter() - t0:.2f}s)')
    return out


def open_pdf(s_or_name):
    """Open a generated PDF in the default viewer (Windows)."""
    name = s_or_name if isinstance(s_or_name, str) else getattr(s_or_name, 'name', 'output')
    os.startfile(os.path.join(PDF_DIR, name + '.pdf'))


def run(sys_or_name, fliess=None, make_pdf=True, verbose=True):
    """approx(...) + pdf(...) in one go."""
    s = approx(sys_or_name, fliess=fliess, verbose=verbose)
    if make_pdf:
        pdf(s)
    return s


def run_all(make_pdf=True, skip=()):
    """Run every example system. Returns {name: ControlSystem or Exception}."""
    results = {}
    for name in SYSTEMS:
        if name in skip:
            continue
        print(f'\n========== {name} ==========')
        try:
            results[name] = run(name, make_pdf=make_pdf, verbose=False)
            print('  OK')
        except Exception as e:  # keep going, report at the end
            results[name] = e
            print(f'  FAILED: {e!r}')
    print('\nSummary:')
    for name, r in results.items():
        print(f"  {name:<22} {'OK' if isinstance(r, ControlSystem) else 'FAILED'}")
    return results


if __name__ == '__main__':
    print(__doc__)
    print('Available systems:')
    ls()
    print('\nTry: s = approx("sine"); plot(s)')
    print()
