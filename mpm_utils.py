import warp as wp
from warp_utils import *
import numpy as np
import math


# compute stress from F
@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)

@wp.func
def kirchoff_stress_water(
    J: float, bulk: float
):
    gamma = 1.1 # gamma is set to be a liitle greater than 1 for weakly compressible fluids
    pressure = -bulk * (wp.pow(J, -gamma) - 1.)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    cauchy_stress = id * pressure
    return J * cauchy_stress

@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    b = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3(
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
    )
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * wp.vec3(
        1.0, 1.0, 1.0
    )
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_StVK(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (
        1.0 / sig[0]
    )
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (
        1.0 / sig[1]
    )
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (
        1.0 / sig[2]
    )
    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)


@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening[mat] > 0.5:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi[mat] * delta_gamma
            )
        return F_elastic
    else:
        return F_trial

@wp.func
def von_mises_return_mapping_with_damage(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        if model.yield_stress[p] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        model.yield_stress[p] = model.yield_stress[p] - model.softening[mat] * wp.length(
            (delta_gamma / epsilon_hat_norm) * epsilon_hat
        )
        if model.yield_stress[p] <= 0:
            model.mu[p] = 0.0
            model.lam[p] = 0.0
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening[mat] > 0.5:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi[mat] * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


# for toothpaste
@wp.func
def viscoplasticity_return_mapping_with_StVK(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, mat: int, dt: float
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    b_trial = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - wp.vec3(
        trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
    )
    s_trial = 2.0 * model.mu[p] * epsilon_hat
    s_trial_norm = wp.length(s_trial)
    y = s_trial_norm - wp.sqrt(2.0 / 3.0) * model.yield_stress[p]
    if y > 0:
        mu_hat = model.mu[p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
        s_new_norm = s_trial_norm - y / (
            1.0 + model.plastic_viscosity[mat] / (2.0 * mu_hat * dt)
        )
        s_new = (s_new_norm / s_trial_norm) * s_trial
        epsilon_new = 1.0 / (2.0 * model.mu[p]) * s_new + wp.vec3(
            trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
        )
        sig_elastic = wp.mat33(
            wp.exp(epsilon_new[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        return F_elastic
    else:
        return F_trial


@wp.func
def sand_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int, mat: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    epsilon = wp.vec3(
        wp.log(wp.max(wp.abs(sig[0]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[1]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[2]), 1e-14)),
    )
    sigma_out = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    epsilon_hat_norm = wp.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * model.lam[p] + 2.0 * model.mu[p])
        / (2.0 * model.mu[p])
        * tr
        * model.alpha[mat]
    )

    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))

        F_elastic = U * wp.diag(s_new) * wp.transpose(V)
    return F_elastic


@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    mat = state.particle_material[p]
    model.mu[p] = model.E[mat] / (2.0 * (1.0 + model.nu[mat]))
    model.lam[p] = model.E[mat] * model.nu[mat] / ((1.0 + model.nu[mat]) * (1.0 - 2.0 * model.nu[mat]))

@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct):
    e, grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[e, grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[e, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx


@wp.func
def update_cov(state: MPMStateStruct, p: int, grad_v: wp.mat33, dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    state.particle_cov[p * 6] = cov_np1[0, 0]
    state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    p = wp.tid()
    e = p // model.n_particles_per_env  # environment index for this particle
    if state.particle_selection[p] == 0:
        stress = state.particle_stress[p]
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    dpos = (
                        wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    ) * model.dx
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    C = state.particle_C[p]
                    # if model.rpic = 0, standard apic
                    C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (
                        C - wp.transpose(C)
                    )
                    if model.rpic_damping < -0.001:
                        # standard pic
                        C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    elastic_force = -state.particle_vol[p] * stress * dweight
                    v_in_add = (
                        weight
                        * state.particle_mass[p]
                        * (state.particle_v[p] + C * dpos)
                        + dt * elastic_force
                    )
                    wp.atomic_add(state.grid_v_in, e, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        state.grid_m, e, ix, iy, iz, weight * state.particle_mass[p]
                    )


# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    e, grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[e, grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[e, grid_x, grid_y, grid_z] * (
            1.0 / state.grid_m[e, grid_x, grid_y, grid_z]
        )
        # per-env gravity (set via set_gravity() or set_gravity_per_env())
        v_out = v_out + dt * model.gravity_per_env[e]
        state.grid_v_out[e, grid_x, grid_y, grid_z] = v_out


@wp.kernel
def apply_external_force_per_env(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    """Apply a per-environment body force to all active, deformable particles.

    Skips inactive particles (particle_selection != 0) and stationary/rigid
    particles (material 7 or 8).  The force is specified in world units (N)
    and is converted to a velocity impulse: dv = force / mass * dt.
    """
    p = wp.tid()
    if state.particle_selection[p] != 0:
        return
    mat = state.particle_material[p]
    if mat == 7 or mat == 8:
        return
    e = p // model.n_particles_per_env
    f = state.external_force_per_env[e]
    # Only apply if the force is non-trivially non-zero (cheap guard)
    if wp.length(f) > 0.0:
        state.particle_v[p] = state.particle_v[p] + f * (dt / state.particle_mass[p])


@wp.kernel
def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    e = p // model.n_particles_per_env  # environment index for this particle
    if state.particle_selection[p] == 0:
        if state.particle_material[p] == 7 or state.particle_material[p] == 8:  # stationary/rigid handled separately
            return
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[e, ix, iy, iz]
                    new_v = new_v + grid_v * weight
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        state.particle_v[p] = new_v
        state.particle_x[p] = state.particle_x[p] + dt * new_v
        state.particle_C[p] = new_C
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_tmp = (I33 + new_F * dt) * state.particle_F[p]
        state.particle_F_trial[p] = F_tmp

        if model.update_cov_with_F:
            update_cov(state, p, new_F, dt)


# compute (Kirchhoff) stress = stress(returnMap(F_trial))
@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        mat = state.particle_material[p]

        # apply return mapping
        if mat == 1:  # metal
            state.particle_F[p] = von_mises_return_mapping(
                state.particle_F_trial[p], model, p, mat
            )
        elif mat == 2:  # sand
            state.particle_F[p] = sand_return_mapping(
                state.particle_F_trial[p], state, model, p, mat
            )
        elif mat == 3:  # visplas, with StVk+VM, no thickening
            state.particle_F[p] = viscoplasticity_return_mapping_with_StVK(
                state.particle_F_trial[p], model, p, mat, dt
            )
        elif mat == 5:  # plasticine
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p, mat
            )
        elif mat == 6:  # fluid
            J = wp.determinant(state.particle_F_trial[p])
            Jcbr = J**(1.0 / 3.0)
            state.particle_F[p] = wp.mat33(Jcbr, 0.0, 0.0, 0.0, Jcbr, 0.0, 0.0, 0.0, Jcbr)
        elif mat == 7 or mat == 8:  # stationary / rigid — no deformation
            state.particle_F[p] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        else:  # jelly (0), snow (4), or custom
            state.particle_F[p] = state.particle_F_trial[p]

        # also compute stress here
        J = wp.determinant(state.particle_F[p])
        U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sig = wp.vec3(0.0)
        stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        wp.svd3(state.particle_F[p], U, sig, V)
        if mat == 0 or mat == 5:
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )
        if mat == 1:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if mat == 2:
            stress = kirchoff_stress_drucker_prager(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if mat == 3:
            # temporarily use stvk, subject to change
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if mat == 6:  # fluid
            stress = kirchoff_stress_water(
                J, model.bulk[mat]
            )

        stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry
        state.particle_stress[p] = stress


@wp.kernel
def compute_cov_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    init_cov = wp.mat33(0.0)
    init_cov[0, 0] = state.particle_init_cov[p * 6]
    init_cov[0, 1] = state.particle_init_cov[p * 6 + 1]
    init_cov[0, 2] = state.particle_init_cov[p * 6 + 2]
    init_cov[1, 0] = state.particle_init_cov[p * 6 + 1]
    init_cov[1, 1] = state.particle_init_cov[p * 6 + 3]
    init_cov[1, 2] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 0] = state.particle_init_cov[p * 6 + 2]
    init_cov[2, 1] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 2] = state.particle_init_cov[p * 6 + 5]

    cov = F * init_cov * wp.transpose(F)

    state.particle_cov[p * 6] = cov[0, 0]
    state.particle_cov[p * 6 + 1] = cov[0, 1]
    state.particle_cov[p * 6 + 2] = cov[0, 2]
    state.particle_cov[p * 6 + 3] = cov[1, 1]
    state.particle_cov[p * 6 + 4] = cov[1, 2]
    state.particle_cov[p * 6 + 5] = cov[2, 2]


@wp.kernel
def compute_R_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    # polar svd decomposition
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    # compute rotation matrix
    R = U * wp.transpose(V)
    state.particle_R[p] = wp.transpose(R)

@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: float):
    e, grid_x, grid_y, grid_z = wp.tid()
    state.grid_v_out[e, grid_x, grid_y, grid_z] = (
        state.grid_v_out[e, grid_x, grid_y, grid_z] * scale
    )


# NOTE: E and nu are now per-type (indexed by material type, not particle).
# To use different E/nu for a region, define a new material type and use
# set_parameters_for_particles() instead.
@wp.kernel
def apply_additional_params(
    state: MPMStateStruct,
    model: MPMModelStruct,
    params_modifier: MaterialParamsModifier,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if (
        pos[0] > params_modifier.point[0] - params_modifier.size[0]
        and pos[0] < params_modifier.point[0] + params_modifier.size[0]
        and pos[1] > params_modifier.point[1] - params_modifier.size[1]
        and pos[1] < params_modifier.point[1] + params_modifier.size[1]
        and pos[2] > params_modifier.point[2] - params_modifier.size[2]
        and pos[2] < params_modifier.point[2] + params_modifier.size[2]
    ):
        state.particle_density[p] = params_modifier.density


@wp.kernel
def selection_add_impulse_on_particles(state: MPMStateStruct, impulse_modifier: Impulse_modifier):
    p = wp.tid()
    offset = state.particle_x[p] - impulse_modifier.point
    if (
        wp.abs(offset[0]) < impulse_modifier.size[0]
        and wp.abs(offset[1]) < impulse_modifier.size[1]
        and wp.abs(offset[2]) < impulse_modifier.size[2]
                ):
        impulse_modifier.mask[p] = 1 
    else:
        impulse_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_translation(state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
                ):
        velocity_modifier.mask[p] = 1 
    else:
        velocity_modifier.mask[p] = 0

        
@wp.kernel
def selection_enforce_particle_velocity_cylinder(state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point

    vertical_distance = wp.abs(wp.dot(offset, velocity_modifier.normal))

    horizontal_distance = wp.length(offset - wp.dot(offset, velocity_modifier.normal) * velocity_modifier.normal)
    if (
        vertical_distance < velocity_modifier.half_height_and_radius[0]
        and horizontal_distance < velocity_modifier.half_height_and_radius[1]
                ):
        velocity_modifier.mask[p] = 1 
    else:
        velocity_modifier.mask[p] = 0


# ---------------------------------------------------------------------------
# Rigid body kernels (material == 8)
# ---------------------------------------------------------------------------

@wp.kernel
def rigid_g2p_accumulate(
    state: MPMStateStruct,
    model: MPMModelStruct,
    rigid_x_cm: wp.array(dtype=wp.vec3),
    rigid_linear_mom: wp.array(dtype=wp.vec3),
    rigid_angular_mom: wp.array(dtype=wp.vec3),
):
    """For each rigid particle gather grid momentum and accumulate per-body
    linear and angular momentum contributions."""
    p = wp.tid()
    e = p // model.n_particles_per_env  # environment index for this particle
    if state.particle_selection[p] == 0 and state.particle_material[p] == 8:
        bid = state.particle_rigid_id[p]

        # standard quadratic B-spline weights (same as g2p)
        grid_pos = state.particle_x[p] * model.inv_dx
        base_x = wp.int(grid_pos[0] - 0.5)
        base_y = wp.int(grid_pos[1] - 0.5)
        base_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(wp.float(base_x), wp.float(base_y), wp.float(base_z))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )

        v_interp = wp.vec3(0.0)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    weight = w[0, i] * w[1, j] * w[2, k]
                    v_interp = v_interp + state.grid_v_out[e, base_x + i, base_y + j, base_z + k] * weight

        b_flat = e * model.n_rigid_bodies_per_env + bid
        mass_p = state.particle_mass[p]
        r = state.particle_x[p] - rigid_x_cm[b_flat]
        wp.atomic_add(rigid_linear_mom, b_flat, v_interp * mass_p)
        wp.atomic_add(rigid_angular_mom, b_flat, wp.cross(r, v_interp * mass_p))


@wp.kernel
def rigid_body_integrate(
    rigid_x_cm: wp.array(dtype=wp.vec3),
    rigid_v_cm: wp.array(dtype=wp.vec3),
    rigid_omega: wp.array(dtype=wp.vec3),
    rigid_orientation: wp.array(dtype=wp.mat33),
    rigid_mass: wp.array(dtype=float),
    rigid_inv_inertia_body: wp.array(dtype=wp.mat33),
    rigid_linear_mom: wp.array(dtype=wp.vec3),
    rigid_angular_mom: wp.array(dtype=wp.vec3),
    dt: float,
):
    """Integrate rigid body EOM for body b (one thread per body).
    Updates v_cm, omega, x_cm, and orientation."""
    b = wp.tid()
    R = rigid_orientation[b]
    M = rigid_mass[b]

    # --- linear ---
    v_cm_new = rigid_linear_mom[b] / M

    # --- angular ---
    # I_world = R * I_body * R^T,  I_world_inv = R * I_body_inv * R^T
    I_world_inv = R * rigid_inv_inertia_body[b] * wp.transpose(R)
    omega_new = I_world_inv * rigid_angular_mom[b]

    # --- position ---
    x_cm_new = rigid_x_cm[b] + v_cm_new * dt

    # --- orientation: first-order update then polar-decomp re-orthogonalisation ---
    ox = omega_new[0]
    oy = omega_new[1]
    oz = omega_new[2]
    skew_w = wp.mat33(0.0, -oz, oy,  oz, 0.0, -ox,  -oy, ox, 0.0)
    R_new = R + skew_w * R * dt
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(R_new, U, sig, V)
    R_orth = U * wp.transpose(V)

    rigid_v_cm[b] = v_cm_new
    rigid_omega[b] = omega_new
    rigid_x_cm[b] = x_cm_new
    rigid_orientation[b] = R_orth


@wp.kernel
def rigid_particle_update(
    state: MPMStateStruct,
    model: MPMModelStruct,
    rigid_x_cm: wp.array(dtype=wp.vec3),
    rigid_v_cm: wp.array(dtype=wp.vec3),
    rigid_omega: wp.array(dtype=wp.vec3),
    rigid_orientation: wp.array(dtype=wp.mat33),
):
    """Set per-particle position, velocity, C, and F from rigid body state."""
    p = wp.tid()
    if state.particle_selection[p] == 0 and state.particle_material[p] == 8:
        bid = state.particle_rigid_id[p]
        e = p // model.n_particles_per_env
        b_flat = e * model.n_rigid_bodies_per_env + bid
        R = rigid_orientation[b_flat]
        x_cm = rigid_x_cm[b_flat]
        v_cm = rigid_v_cm[b_flat]
        omega = rigid_omega[b_flat]

        x_p = x_cm + R * state.particle_x_ref[p]
        r = x_p - x_cm

        # velocity: v_cm + omega x r
        state.particle_v[p] = v_cm + wp.cross(omega, r)
        state.particle_x[p] = x_p

        # C = skew(omega) so APIC scatter uses the correct linear velocity field
        ox = omega[0]
        oy = omega[1]
        oz = omega[2]
        state.particle_C[p] = wp.mat33(0.0, -oz, oy,  oz, 0.0, -ox,  -oy, ox, 0.0)

        # rigid — no deformation
        state.particle_F[p] = wp.mat33(1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0)


@wp.kernel
def rigid_collide_planes_kernel(
    particle_x_ref: wp.array(dtype=wp.vec3),
    rigid_particle_indices: wp.array(dtype=int),
    rigid_body_particle_start: wp.array(dtype=int),
    rigid_body_particle_end: wp.array(dtype=int),
    rigid_x_cm: wp.array(dtype=wp.vec3),
    rigid_v_cm: wp.array(dtype=wp.vec3),
    rigid_omega: wp.array(dtype=wp.vec3),
    rigid_orientation: wp.array(dtype=wp.mat33),
    rigid_mass: wp.array(dtype=float),
    rigid_inv_inertia_body: wp.array(dtype=wp.mat33),
    surf_point: wp.array(dtype=wp.vec3),
    surf_normal: wp.array(dtype=wp.vec3),
    surf_restitution: wp.array(dtype=float),
    surf_friction: wp.array(dtype=float),
    surf_start_time: wp.array(dtype=float),
    surf_end_time: wp.array(dtype=float),
    n_surfaces: int,
    current_time: float,
    dt: float,
):
    """One thread per rigid body (across all envs).

    Uses a pre-built compact index list (rigid_particle_indices) so the inner
    loop only iterates over the few rigid particles belonging to each body,
    not over all particles in the env.

    Impulses from multiple surfaces are accumulated in local registers and
    written back once — no atomics needed."""
    b_flat = wp.tid()

    x_cm        = rigid_x_cm[b_flat]
    R           = rigid_orientation[b_flat]
    M           = rigid_mass[b_flat]
    I_inv       = rigid_inv_inertia_body[b_flat]
    I_world_inv = R * I_inv * wp.transpose(R)

    # Read current velocities into local vars; accumulate all surface impulses
    v_cm  = rigid_v_cm[b_flat]
    omega = rigid_omega[b_flat]

    k_start = rigid_body_particle_start[b_flat]
    k_end   = rigid_body_particle_end[b_flat]

    for s in range(n_surfaces):
        if current_time < surf_start_time[s] or current_time >= surf_end_time[s]:
            continue

        pt      = surf_point[s]
        n_hat   = surf_normal[s]
        coeff_e = surf_restitution[s]
        mu      = surf_friction[s]

        # Find the deepest penetrating rigid particle for this body
        min_dist    = float(1.0e10)
        min_x_world = wp.vec3(0.0, 0.0, 0.0)

        for k in range(k_start, k_end):
            p       = rigid_particle_indices[k]
            x_world = x_cm + R * particle_x_ref[p]
            dist    = wp.dot(x_world - pt, n_hat)
            if dist < min_dist:
                min_dist    = dist
                min_x_world = x_world

        if min_dist >= 0.0:
            continue  # no penetration

        r         = min_x_world - x_cm
        v_contact = v_cm + wp.cross(omega, r)
        v_n       = wp.dot(v_contact, n_hat)

        if v_n >= 0.0:
            continue  # already separating

        r_cross_n = wp.cross(r, n_hat)
        denom     = (1.0 / M) + wp.dot(wp.cross(I_world_inv * r_cross_n, r), n_hat)
        J_n       = -(1.0 + coeff_e) * v_n / denom

        v_cm  = v_cm  + n_hat * (J_n / M)
        omega = omega + I_world_inv * r_cross_n * J_n

        # Coulomb friction impulse
        if mu > 0.0:
            v_t_vec = v_contact - v_n * n_hat
            v_t_mag = wp.length(v_t_vec)
            if v_t_mag > 1.0e-10:
                t_hat     = v_t_vec * (1.0 / v_t_mag)
                r_cross_t = wp.cross(r, t_hat)
                denom_t   = (1.0 / M) + wp.dot(wp.cross(I_world_inv * r_cross_t, r), t_hat)
                J_t       = wp.min(v_t_mag / denom_t, mu * J_n)
                v_cm  = v_cm  - t_hat * (J_t / M)
                omega = omega - I_world_inv * r_cross_t * J_t

    # Write accumulated velocities back once — one thread per body, no races
    rigid_v_cm[b_flat]  = v_cm
    rigid_omega[b_flat] = omega


# ---------------------------------------------------------------------------
# SDF collider helpers and kernel
# ---------------------------------------------------------------------------

@wp.func
def sdf_sample(sdf: wp.array(dtype=float, ndim=3),
               nx: int, ny: int, nz: int,
               i: int, j: int, k: int) -> float:
    """Nearest-neighbour SDF lookup with boundary clamping."""
    i = wp.clamp(i, 0, nx - 1)
    j = wp.clamp(j, 0, ny - 1)
    k = wp.clamp(k, 0, nz - 1)
    return sdf[i, j, k]


@wp.func
def sdf_query(sdf: wp.array(dtype=float, ndim=3),
              origin: wp.vec3, dx_sdf: float,
              nx: int, ny: int, nz: int,
              world_pos: wp.vec3) -> float:
    """Trilinearly interpolated SDF value at world_pos (in SDF local frame)."""
    p  = (world_pos - origin) / dx_sdf

    # Out-of-bounds check on continuous coords: if the position is outside
    # [origin, origin + (n-1)*dx_sdf] in any axis, return large positive phi
    # so no collision is triggered.  We check continuous p here (before int())
    # because int() in Warp truncates toward zero, so p[0] in [-1, 0) gives
    # i=0 and a negative fractional weight, causing extrapolation rather than
    # interpolation and producing phi < boundary value.
    if p[0] < 0.0 or p[0] >= float(nx - 1) or p[1] < 0.0 or p[1] >= float(ny - 1) or p[2] < 0.0 or p[2] >= float(nz - 1):
        return float(1.0e10)

    i  = int(p[0]);  j  = int(p[1]);  k  = int(p[2])

    fx = p[0] - float(i)
    fy = p[1] - float(j)
    fz = p[2] - float(k)

    c000 = sdf_sample(sdf, nx, ny, nz, i,   j,   k  )
    c100 = sdf_sample(sdf, nx, ny, nz, i+1, j,   k  )
    c010 = sdf_sample(sdf, nx, ny, nz, i,   j+1, k  )
    c110 = sdf_sample(sdf, nx, ny, nz, i+1, j+1, k  )
    c001 = sdf_sample(sdf, nx, ny, nz, i,   j,   k+1)
    c101 = sdf_sample(sdf, nx, ny, nz, i+1, j,   k+1)
    c011 = sdf_sample(sdf, nx, ny, nz, i,   j+1, k+1)
    c111 = sdf_sample(sdf, nx, ny, nz, i+1, j+1, k+1)

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0  = c00  * (1.0 - fy) + c10  * fy
    c1  = c01  * (1.0 - fy) + c11  * fy
    return c0 * (1.0 - fz) + c1 * fz


@wp.func
def sdf_gradient(sdf: wp.array(dtype=float, ndim=3),
                 origin: wp.vec3, dx_sdf: float,
                 nx: int, ny: int, nz: int,
                 local_pos: wp.vec3) -> wp.vec3:
    """Central-difference gradient of SDF → outward surface normal (unit vector)."""
    ex = wp.vec3(dx_sdf, 0.0, 0.0)
    ey = wp.vec3(0.0, dx_sdf, 0.0)
    ez = wp.vec3(0.0, 0.0, dx_sdf)
    gx = (sdf_query(sdf, origin, dx_sdf, nx, ny, nz, local_pos + ex) -
          sdf_query(sdf, origin, dx_sdf, nx, ny, nz, local_pos - ex)) / (2.0 * dx_sdf)
    gy = (sdf_query(sdf, origin, dx_sdf, nx, ny, nz, local_pos + ey) -
          sdf_query(sdf, origin, dx_sdf, nx, ny, nz, local_pos - ey)) / (2.0 * dx_sdf)
    gz = (sdf_query(sdf, origin, dx_sdf, nx, ny, nz, local_pos + ez) -
          sdf_query(sdf, origin, dx_sdf, nx, ny, nz, local_pos - ez)) / (2.0 * dx_sdf)
    return wp.normalize(wp.vec3(gx, gy, gz))


@wp.kernel
def collide_sdf(
    time:  float,
    dt:    float,
    state: MPMStateStruct,
    model: MPMModelStruct,
    param: SDFCollider,
):
    """Grid-level SDF collision kernel.

    For each grid node, transforms the node's world position into the
    collider's local frame, queries the SDF, and applies sticky / slip /
    frictional velocity correction when the node is inside the object (φ < 0).
    """
    e, gx, gy, gz = wp.tid()
    if time < param.start_time or time >= param.end_time:
        return

    # World position of this grid node (same domain for all envs)
    world_pos = wp.vec3(float(gx) * model.dx,
                        float(gy) * model.dx,
                        float(gz) * model.dx)

    # Transform to SDF local frame: local = R^T (world - translation)
    R_inv     = wp.transpose(param.rotation)
    local_pos = R_inv * (world_pos - param.translation)

    phi = sdf_query(param.sdf, param.origin, param.dx_sdf,
                    param.nx, param.ny, param.nz, local_pos)

    margin = param.margin

    if phi < margin:
        # Outward normal in local frame → rotate to world frame
        n_local = sdf_gradient(param.sdf, param.origin, param.dx_sdf,
                               param.nx, param.ny, param.nz, local_pos)
        n = wp.normalize(param.rotation * n_local)

        v  = state.grid_v_out[e, gx, gy, gz]
        vn = wp.dot(v, n)

        # Surface velocity of the collider at this grid node:
        #   v_surf = v_linear + ω × (world_pos - translation)
        r = world_pos - param.translation
        v_surf = param.linear_velocity + wp.cross(param.angular_velocity, r)
        vn_surf = wp.dot(v_surf, n)

        if phi < 0.0:
            # Fully inside solid: apply surface-type BC.
            if param.surface_type == 0:
                # Sticky: match surface velocity exactly
                state.grid_v_out[e, gx, gy, gz] = v_surf

            elif param.surface_type == 1:
                # Slip: enforce v·n >= v_surf·n, keep tangential component of grid velocity
                if vn < vn_surf:
                    state.grid_v_out[e, gx, gy, gz] = v + (vn_surf - vn) * n

            else:
                # Frictional: enforce normal, apply Coulomb friction on tangential relative velocity
                v_rel = v - v_surf
                vn_rel = wp.dot(v_rel, n)
                if vn_rel < 0.0:
                    v_rel_t = v_rel - vn_rel * n   # tangential relative velocity (normal already removed)
                    v_rel_t_len = wp.length(v_rel_t)
                    if v_rel_t_len > 1e-20:
                        friction_impulse = wp.min(param.friction * wp.abs(vn_rel), v_rel_t_len)
                        v_rel_t = v_rel_t * (1.0 - friction_impulse / v_rel_t_len)
                    v = v_surf + v_rel_t  # surface velocity + corrected tangential relative velocity
                    state.grid_v_out[e, gx, gy, gz] = v

        else:
            # Margin zone (0 <= φ < margin): only push if surface is moving outward faster than grid.
            if vn < vn_surf:
                state.grid_v_out[e, gx, gy, gz] = v + (vn_surf - vn) * n
