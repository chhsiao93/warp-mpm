import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from engine_utils import *
from warp_utils import *
from mpm_utils import *


MATERIAL_NAME_TO_ID = {
    "jelly": 0, "metal": 1, "sand": 2, "foam": 3,
    "snow": 4, "plasticine": 5, "fluid": 6, "stationary": 7, "rigid": 8,
}
MAX_MATERIALS = 16


class MPM_Simulator_WARP:
    def __init__(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda:0", n_envs=1):
        self.initialize(n_particles, n_grid, grid_lim, device=device, n_envs=n_envs)
        self.time_profile = {}

    def initialize(self, n_particles, n_grid=100, grid_lim=1.0, device="cuda:0", n_envs=1):
        self.n_particles = n_particles
        # Multi-env support: n_particles is the TOTAL across all environments.
        # n_particles_per_env = n_particles // n_envs (must divide evenly).
        assert n_particles % n_envs == 0, \
            f"n_particles ({n_particles}) must be divisible by n_envs ({n_envs})"
        self.n_envs = n_envs
        self.n_particles_per_env = n_particles // n_envs

        self.mpm_model = MPMModelStruct()
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        self.mpm_model.grid_lim = grid_lim
        self.mpm_model.n_grid = n_grid
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        self.mpm_model.n_envs = n_envs
        self.mpm_model.n_particles_per_env = self.n_particles_per_env
        self.mpm_model.n_rigid_bodies_per_env = 0  # set by initialize_rigid_bodies
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )

        # per-type material parameters (indexed by material type, not particle)
        self.mpm_model.E = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.nu = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.bulk = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.hardening = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.xi = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.plastic_viscosity = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.softening = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)
        self.mpm_model.alpha = wp.zeros(shape=MAX_MATERIALS, dtype=float, device=device)

        # per-particle material parameters (can evolve during simulation)
        self.mpm_model.mu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.lam = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.yield_stress = wp.zeros(shape=n_particles, dtype=float, device=device)

        self.mpm_model.update_cov_with_F = False

        # material is used to switch between different elastoplastic models. 0 is jelly
        self.mpm_model.material = 0

        # default friction_angle and alpha for all types
        self.mpm_model.friction_angle = 25.0
        sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
        default_alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
        wp.launch(kernel=set_value_to_float_array, dim=MAX_MATERIALS,
                  inputs=[self.mpm_model.alpha, default_alpha], device=device)
        wp.launch(kernel=set_value_to_float_array, dim=MAX_MATERIALS,
                  inputs=[self.mpm_model.softening, 0.1], device=device)

        self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)

        # per-env gravity — initialised to zero; broadcast via set_gravity() or
        # per-env via set_gravity_per_env().
        self.mpm_model.gravity_per_env = wp.zeros(shape=n_envs, dtype=wp.vec3, device=device)

        self.mpm_model.rpic_damping = 0.0  # 0.0 if no damping (apic). -1 if pic

        self.mpm_model.grid_v_damping_scale = 1.1  # globally applied

        self.mpm_state = MPMStateStruct()

        self.mpm_state.particle_x = wp.empty(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # current position

        self.mpm_state.particle_v = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity

        self.mpm_state.particle_F = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle F elastic

        self.mpm_state.particle_R = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle R rotation

        self.mpm_state.particle_init_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # initial covariance matrix

        self.mpm_state.particle_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # current covariance matrix

        self.mpm_state.particle_F_trial = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # apply return mapping will yield

        self.mpm_state.particle_stress = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )

        self.mpm_state.particle_vol = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle volume
        self.mpm_state.particle_mass = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle mass
        self.mpm_state.particle_density = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_state.particle_C = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )
        self.mpm_state.particle_Jp = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )

        self.mpm_state.particle_selection = wp.zeros(
            shape=n_particles, dtype=int, device=device
        )

        self.mpm_state.particle_material = wp.zeros(
            shape=n_particles, dtype=int, device=device
        )  # default 0 = jelly

        self.mpm_state.particle_rigid_id = wp.zeros(
            shape=n_particles, dtype=int, device=device
        )  # rigid body id; only meaningful when particle_material == 8
        self.mpm_state.particle_x_ref = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # body-frame reference position for rigid particles

        # per-env external body force (RL actions); zero until explicitly set
        self.mpm_state.external_force_per_env = wp.zeros(
            shape=n_envs, dtype=wp.vec3, device=device
        )

        _g = self.mpm_model.n_grid
        self.mpm_state.grid_m = wp.zeros(
            shape=(n_envs, _g, _g, _g),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(n_envs, _g, _g, _g),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(n_envs, _g, _g, _g),
            dtype=wp.vec3,
            device=device,
        )

        self.time = 0.0

        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []

        self.tailored_struct_for_bc = MPMtailoredStruct()
        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []

        # surfaces that apply restitution impulses to rigid bodies
        self.rigid_surface_colliders = []

        # rigid body state (populated by initialize_rigid_bodies / finalize_rigid_bodies)
        self.n_rigid_bodies = 0
        self.n_rigid_bodies_per_env = 0
        # Compact per-body particle index list (built in finalize_rigid_bodies)
        self._rigid_particle_indices = None      # flat array: all rigid particle global indices
        self._rigid_body_particle_start = None   # start index in _rigid_particle_indices for each body
        self._rigid_body_particle_end = None     # end index (exclusive) for each body
        # GPU copies of rigid surface collider data (uploaded once in _upload_rigid_surface_colliders)
        self._rigid_surf_uploaded = False        # True once the GPU arrays match rigid_surface_colliders
        self._rigid_surf_points = None           # surf point positions on GPU
        self.rigid_x_cm = None
        self.rigid_v_cm = None
        self.rigid_omega = None
        self.rigid_orientation = None
        self.rigid_mass = None
        self.rigid_inv_inertia_body = None
        self._rigid_linear_mom = None   # per-step accumulation buffer
        self._rigid_angular_mom = None  # per-step accumulation buffer

    # the h5 file should store particle initial position and volume.
    def load_from_sampling(
        self, sampling_h5, n_grid=100, grid_lim=1.0, device="cuda:0"
    ):
        if not os.path.exists(sampling_h5):
            print("h5 file cannot be found at ", os.getcwd() + sampling_h5)
            exit()

        h5file = h5py.File(sampling_h5, "r")
        x, particle_volume = h5file["x"], h5file["particle_volume"]

        x = x[()].transpose()  # np vector of x # shape now is (n_particles, dim)

        self.dim, self.n_particles = x.shape[1], x.shape[0]

        self.initialize(self.n_particles, n_grid, grid_lim, device=device)

        print(
            "Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particles"
        )
        particle_volume = np.squeeze(particle_volume, 0)

        self.mpm_state.particle_x = wp.from_numpy(
            x, dtype=wp.vec3, device=device
        )  # initialize warp array from np

        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        # initial deformation gradient is set to identity

        self.mpm_state.particle_vol = wp.from_numpy(
            particle_volume, dtype=float, device=device
        )

        print("Particles initialized from sampling file.")
        print("Total particles: ", self.n_particles)

    # shape of tensor_x is (n, 3); shape of tensor_volume is (n,)
    # n is the TOTAL particle count across all environments (= n_envs * n_particles_per_env)
    def load_initial_data_from_torch(
        self,
        tensor_x,
        tensor_volume,
        tensor_cov = None,
        n_grid=100,
        grid_lim=1.0,
        device="cuda:0",
        n_envs=1,
    ):
        self.dim, self.n_particles = tensor_x.shape[1], tensor_x.shape[0]
        assert tensor_x.shape[0] == tensor_volume.shape[0]
        # assert tensor_x.shape[0] == tensor_cov.reshape(-1, 6).shape[0]
        self.initialize(self.n_particles, n_grid, grid_lim, device=device, n_envs=n_envs)

        self.import_particle_x_from_torch(tensor_x, device)
        self.mpm_state.particle_vol = wp.from_numpy(
            tensor_volume.detach().clone().cpu().numpy(), dtype=float, device=device
        )
        if tensor_cov is not None:
            self.mpm_state.particle_init_cov = wp.from_numpy(
                tensor_cov.reshape(-1).detach().clone().cpu().numpy(),
                dtype=float,
                device=device,
            )

            if self.mpm_model.update_cov_with_F:
                self.mpm_state.particle_cov = self.mpm_state.particle_init_cov

        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        # initial trial deformation gradient is set to identity

        print("Particles initialized from torch data.")
        print("Total particles: ", self.n_particles)

    # must give density. mass will be updated as density * volume
    def set_parameters(self, device="cuda:0", **kwargs):
        self.set_parameters_dict(device, kwargs)

    def _material_name_to_id(self, name):
        if isinstance(name, int):
            return name
        if name not in MATERIAL_NAME_TO_ID:
            raise TypeError(f"Undefined material type: {name}")
        return MATERIAL_NAME_TO_ID[name]

    def set_parameters_dict(self, kwargs={}, device="cuda:0"):
        if "material" in kwargs:
            mat_id = self._material_name_to_id(kwargs["material"])
            self.mpm_model.material = mat_id
            # broadcast material type to all particles
            wp.launch(
                kernel=set_value_to_int_array,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_material, mat_id],
                device=device,
            )

        if "grid_lim" in kwargs:
            self.mpm_model.grid_lim = kwargs["grid_lim"]
        if "n_grid" in kwargs:
            self.mpm_model.n_grid = kwargs["n_grid"]
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )
        _g = self.mpm_model.n_grid
        self.mpm_state.grid_m = wp.zeros(
            shape=(self.n_envs, _g, _g, _g),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.n_envs, _g, _g, _g),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.n_envs, _g, _g, _g),
            dtype=wp.vec3,
            device=device,
        )

        mat_id = self.mpm_model.material  # current material type for per-type indexing

        # per-type parameters (E, nu, bulk written to model arrays at mat_id index)
        if "E" in kwargs:
            E_torch = wp.to_torch(self.mpm_model.E)
            E_torch[mat_id] = kwargs["E"]
        if "nu" in kwargs:
            nu_torch = wp.to_torch(self.mpm_model.nu)
            nu_torch[mat_id] = kwargs["nu"]
        if "bulk_modulus" in kwargs:
            bulk_torch = wp.to_torch(self.mpm_model.bulk)
            bulk_torch[mat_id] = kwargs["bulk_modulus"]

        # per-particle parameters
        if "yield_stress" in kwargs:
            val = kwargs["yield_stress"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.yield_stress, val],
                device=device,
            )

        # per-type plasticity parameters
        if "hardening" in kwargs:
            h_torch = wp.to_torch(self.mpm_model.hardening)
            h_torch[mat_id] = float(kwargs["hardening"])
        if "xi" in kwargs:
            xi_torch = wp.to_torch(self.mpm_model.xi)
            xi_torch[mat_id] = kwargs["xi"]
        if "friction_angle" in kwargs:
            self.mpm_model.friction_angle = kwargs["friction_angle"]
            sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
            alpha_val = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
            alpha_torch = wp.to_torch(self.mpm_model.alpha)
            alpha_torch[mat_id] = alpha_val
        if "plastic_viscosity" in kwargs:
            pv_torch = wp.to_torch(self.mpm_model.plastic_viscosity)
            pv_torch[mat_id] = kwargs["plastic_viscosity"]
        if "softening" in kwargs:
            s_torch = wp.to_torch(self.mpm_model.softening)
            s_torch[mat_id] = kwargs["softening"]

        if "g" in kwargs:
            self.set_gravity(kwargs["g"])

        if "density" in kwargs:
            density_value = kwargs["density"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_density, density_value],
                device=device,
            )
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )

        # compute per-particle mu/lam from per-type E/nu if E or nu was set
        if "E" in kwargs or "nu" in kwargs:
            wp.launch(
                kernel=compute_mu_lam_from_E_nu,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )

        if "rpic_damping" in kwargs:
            self.mpm_model.rpic_damping = kwargs["rpic_damping"]
        if "grid_v_damping_scale" in kwargs:
            self.mpm_model.grid_v_damping_scale = kwargs["grid_v_damping_scale"]

        if "additional_material_params" in kwargs:
            for params in kwargs["additional_material_params"]:
                param_modifier = MaterialParamsModifier()
                param_modifier.point = wp.vec3(params["point"])
                param_modifier.size = wp.vec3(params["size"])
                param_modifier.density = params["density"]
                wp.launch(
                    kernel=apply_additional_params,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, param_modifier],
                    device=device,
                )

            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )


    def finalize_mu_lam(self, device="cuda:0"):
        wp.launch(kernel=compute_mu_lam_from_E_nu, dim=self.n_particles,
                  inputs=[self.mpm_state, self.mpm_model], device=device)

    # kept for backward compatibility
    def finalize_mu_lam_bulk(self, device="cuda:0"):
        self.finalize_mu_lam(device=device)

    def set_gravity(self, g):
        """Set the same gravity for all environments (backward-compatible)."""
        gv = wp.vec3(float(g[0]), float(g[1]), float(g[2]))
        self.mpm_model.gravitational_accelaration = gv  # kept for reference
        g_torch = wp.to_torch(self.mpm_model.gravity_per_env)
        g_torch[:] = torch.tensor([g[0], g[1], g[2]], dtype=torch.float32,
                                   device=g_torch.device)

    def set_gravity_per_env(self, env_idx: int, g):
        """Set gravity for a single environment (0-indexed).

        Args:
            env_idx: environment index in [0, n_envs)
            g: sequence of 3 floats [gx, gy, gz]
        """
        assert 0 <= env_idx < self.n_envs, f"env_idx {env_idx} out of range [0, {self.n_envs})"
        g_torch = wp.to_torch(self.mpm_model.gravity_per_env)
        g_torch[env_idx] = torch.tensor([g[0], g[1], g[2]], dtype=torch.float32,
                                         device=g_torch.device)

    def set_gravity_all_envs(self, g_tensor):
        """Set gravity for all environments from a (E, 3) or (3,) tensor.

        Args:
            g_tensor: torch.Tensor, shape (E, 3) for per-env gravity,
                      or shape (3,) to broadcast to all envs.
        """
        g_tensor = g_tensor.float()
        if g_tensor.ndim == 1:
            g_tensor = g_tensor.unsqueeze(0).expand(self.n_envs, 3).contiguous()
        assert g_tensor.shape == (self.n_envs, 3), \
            f"Expected shape ({self.n_envs}, 3), got {g_tensor.shape}"
        g_torch = wp.to_torch(self.mpm_model.gravity_per_env)
        g_torch[:] = g_tensor.to(g_torch.device)

    # ------------------------------------------------------------------
    # Per-env external force API  (RL actions)
    # ------------------------------------------------------------------

    def set_env_external_force(self, env_idx: int, force, device="cuda:0"):
        """Set a body force (N) applied to all active deformable particles in env_idx.

        The force persists until changed.  Call with force=[0,0,0] to clear.

        Args:
            env_idx: environment index in [0, n_envs)
            force: sequence of 3 floats [fx, fy, fz] in Newtons
        """
        assert 0 <= env_idx < self.n_envs, f"env_idx {env_idx} out of range [0, {self.n_envs})"
        f_torch = wp.to_torch(self.mpm_state.external_force_per_env)
        f_torch[env_idx] = torch.tensor([force[0], force[1], force[2]],
                                         dtype=torch.float32, device=f_torch.device)

    def set_env_external_force_all(self, force_tensor, device="cuda:0"):
        """Set external body forces for all environments at once.

        Typical RL usage — call this every step before p2g2p() with the
        policy action output:

            solver.set_env_external_force_all(actions)  # (E, 3)
            solver.p2g2p(step, dt)

        Args:
            force_tensor: torch.Tensor shape (E, 3) — force in Newtons per env,
                          or shape (3,) to broadcast to all envs.
        """
        force_tensor = force_tensor.float()
        if force_tensor.ndim == 1:
            force_tensor = force_tensor.unsqueeze(0).expand(self.n_envs, 3).contiguous()
        assert force_tensor.shape == (self.n_envs, 3), \
            f"Expected shape ({self.n_envs}, 3), got {force_tensor.shape}"
        f_torch = wp.to_torch(self.mpm_state.external_force_per_env)
        f_torch[:] = force_tensor.to(f_torch.device)

    def clear_env_external_force(self, device="cuda:0"):
        """Zero all per-env external forces."""
        wp.launch(kernel=set_vec3_to_zero,
                  dim=self.n_envs,
                  inputs=[self.mpm_state.external_force_per_env],
                  device=device)

    # ------------------------------------------------------------------
    # Rigid body API
    # ------------------------------------------------------------------

    def initialize_rigid_bodies(self, n_bodies_per_env, device="cuda:0"):
        """Allocate per-body state arrays for n_bodies_per_env rigid bodies per environment.
        Total body arrays have shape (n_envs * n_bodies_per_env,).
        Call this before finalize_rigid_bodies()."""
        self.n_rigid_bodies_per_env = n_bodies_per_env
        self.n_rigid_bodies = self.n_envs * n_bodies_per_env
        self.mpm_model.n_rigid_bodies_per_env = n_bodies_per_env
        n_total = self.n_rigid_bodies
        self.rigid_x_cm = wp.zeros(n_total, dtype=wp.vec3, device=device)
        self.rigid_v_cm = wp.zeros(n_total, dtype=wp.vec3, device=device)
        self.rigid_omega = wp.zeros(n_total, dtype=wp.vec3, device=device)
        self.rigid_orientation = wp.from_numpy(
            np.tile(np.eye(3, dtype=np.float32), (n_total, 1, 1)).reshape(n_total, 3, 3),
            dtype=wp.mat33, device=device,
        )
        self.rigid_mass = wp.zeros(n_total, dtype=float, device=device)
        self.rigid_inv_inertia_body = wp.zeros(n_total, dtype=wp.mat33, device=device)
        self._rigid_linear_mom = wp.zeros(n_total, dtype=wp.vec3, device=device)
        self._rigid_angular_mom = wp.zeros(n_total, dtype=wp.vec3, device=device)

    def finalize_rigid_bodies(self, device="cuda:0"):
        """Compute center of mass, inertia tensor, and reference positions for
        all rigid bodies from the current particle layout.  Must be called after
        all set_parameters_for_particles() calls that assign material="rigid" and
        obj_id, and before the first p2g2p() step.

        particle_rigid_id[p] stores the LOCAL body id within its environment (0-based).
        Body arrays are laid out flat as [env0_body0, env0_body1, ..., env1_body0, ...].
        """
        P = self.n_particles_per_env
        x_np = self.mpm_state.particle_x.numpy()          # (E*P, 3)
        m_np = self.mpm_state.particle_mass.numpy()        # (E*P,)
        mat_np = self.mpm_state.particle_material.numpy()  # (E*P,)
        rid_np = self.mpm_state.particle_rigid_id.numpy()  # (E*P,) — local body id per env

        rigid_mask = mat_np == 8
        if not np.any(rigid_mask):
            return

        # auto-detect n_bodies_per_env from env 0 (all envs must have the same layout)
        env0_mask = rigid_mask[:P]
        n_bodies_per_env = int(rid_np[:P][env0_mask].max()) + 1 if np.any(env0_mask) else 0
        if n_bodies_per_env == 0:
            return
        self.initialize_rigid_bodies(n_bodies_per_env, device=device)

        n_total = self.n_rigid_bodies  # = n_envs * n_bodies_per_env
        x_cm_np = np.zeros((n_total, 3), dtype=np.float32)
        mass_np = np.zeros(n_total, dtype=np.float32)
        I_body_inv_np = np.zeros((n_total, 3, 3), dtype=np.float32)
        x_ref_np = np.zeros_like(x_np)

        for e in range(self.n_envs):
            p_start = e * P
            p_end = p_start + P
            mat_e = mat_np[p_start:p_end]
            rid_e = rid_np[p_start:p_end]
            x_e = x_np[p_start:p_end]
            m_e = m_np[p_start:p_end]

            for b_local in range(n_bodies_per_env):
                b_flat = e * n_bodies_per_env + b_local
                idx = np.where((mat_e == 8) & (rid_e == b_local))[0]
                if len(idx) == 0:
                    continue
                m_b = m_e[idx]
                mass_np[b_flat] = float(m_b.sum())
                x_cm_np[b_flat] = (x_e[idx] * m_b[:, None]).sum(axis=0) / mass_np[b_flat]

                # inertia tensor in body frame (= world frame at t=0, orientation = I)
                r = x_e[idx] - x_cm_np[b_flat]
                x_ref_np[p_start + idx] = r  # body-frame reference positions
                I = np.zeros((3, 3), dtype=np.float64)
                for ri, mi in zip(r, m_b):
                    I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
                I_body_inv_np[b_flat] = np.linalg.inv(I).astype(np.float32)

        # push rigid body state to GPU
        self.rigid_x_cm = wp.from_numpy(x_cm_np, dtype=wp.vec3, device=device)
        self.rigid_mass = wp.from_numpy(mass_np, dtype=float, device=device)
        self.rigid_inv_inertia_body = wp.from_numpy(
            I_body_inv_np.reshape(n_total, 3, 3), dtype=wp.mat33, device=device
        )
        self.mpm_state.particle_x_ref = wp.from_numpy(x_ref_np, dtype=wp.vec3, device=device)

        # Build compact per-body particle index lists for the GPU collision kernel.
        # Iterating only over rigid particles (instead of all env particles) keeps
        # the kernel fast even for single-env simulations.
        all_indices = []
        body_p_start_np = np.zeros(n_total, dtype=np.int32)
        body_p_end_np   = np.zeros(n_total, dtype=np.int32)
        for e in range(self.n_envs):
            p_start_e = e * P
            mat_e = mat_np[p_start_e : p_start_e + P]
            rid_e = rid_np[p_start_e : p_start_e + P]
            for b_local in range(n_bodies_per_env):
                b_flat = e * n_bodies_per_env + b_local
                local_idx = np.where((mat_e == 8) & (rid_e == b_local))[0]
                global_idx = (local_idx + p_start_e).astype(np.int32)
                body_p_start_np[b_flat] = len(all_indices)
                all_indices.extend(global_idx.tolist())
                body_p_end_np[b_flat] = len(all_indices)

        indices_np = np.array(all_indices, dtype=np.int32) if all_indices else np.zeros(1, dtype=np.int32)
        self._rigid_particle_indices    = wp.from_numpy(indices_np,       dtype=int, device=device)
        self._rigid_body_particle_start = wp.from_numpy(body_p_start_np,  dtype=int, device=device)
        self._rigid_body_particle_end   = wp.from_numpy(body_p_end_np,    dtype=int, device=device)

    def set_rigid_body_velocity(self, body_id, v_cm=(0, 0, 0), omega=(0, 0, 0), env_id=0, device="cuda:0"):
        """Set the linear and angular velocity of a rigid body and immediately
        propagate the new velocity to the body's particles so the next P2G step
        carries the correct momentum.

        body_id: LOCAL body id within the environment (0-based).
        env_id:  which environment to target (default 0)."""
        b_flat  = env_id * self.n_rigid_bodies_per_env + body_id
        v_t = wp.to_torch(self.rigid_v_cm)
        o_t = wp.to_torch(self.rigid_omega)
        v_t[b_flat] = torch.tensor(v_cm,   dtype=torch.float32, device=v_t.device)
        o_t[b_flat] = torch.tensor(omega,  dtype=torch.float32, device=o_t.device)
        # Propagate new velocity to per-particle arrays so that P2G in the next
        # step scatters the correct momentum to the grid.
        self.sync_rigid_particle_state(device=device)

    def _upload_rigid_surface_colliders(self, device):
        """Upload rigid surface collider data to GPU once.
        Called automatically before the first simulation step that needs it."""
        surfs = self.rigid_surface_colliders
        if not surfs:
            return
        pts   = np.array([s["point"]       for s in surfs], dtype=np.float32)
        nors  = np.array([s["normal"]      for s in surfs], dtype=np.float32)
        rest  = np.array([s["restitution"] for s in surfs], dtype=np.float32)
        fric  = np.array([s["friction"]    for s in surfs], dtype=np.float32)
        t0    = np.array([s["start_time"]  for s in surfs], dtype=np.float32)
        t1    = np.array([s["end_time"]    for s in surfs], dtype=np.float32)
        self._rigid_surf_points      = wp.from_numpy(pts,  dtype=wp.vec3, device=device)
        self._rigid_surf_normals     = wp.from_numpy(nors, dtype=wp.vec3, device=device)
        self._rigid_surf_restitutions = wp.from_numpy(rest, dtype=float,  device=device)
        self._rigid_surf_frictions   = wp.from_numpy(fric, dtype=float,  device=device)
        self._rigid_surf_start_times = wp.from_numpy(t0,   dtype=float,  device=device)
        self._rigid_surf_end_times   = wp.from_numpy(t1,   dtype=float,  device=device)
        self._rigid_surf_uploaded = True

    def _apply_rigid_restitution(self, dt, device="cuda:0"):
        """Apply restitution impulses to rigid bodies via GPU kernel.
        Surface data is uploaded to GPU once on first call and reused every step."""
        if not self.rigid_surface_colliders or self.n_rigid_bodies == 0:
            return
        if not self._rigid_surf_uploaded:
            self._upload_rigid_surface_colliders(device)
        wp.launch(
            kernel=rigid_collide_planes_kernel,
            dim=self.n_rigid_bodies,
            inputs=[
                self.mpm_state.particle_x_ref,
                self._rigid_particle_indices,
                self._rigid_body_particle_start,
                self._rigid_body_particle_end,
                self.rigid_x_cm, self.rigid_v_cm, self.rigid_omega,
                self.rigid_orientation, self.rigid_mass, self.rigid_inv_inertia_body,
                self._rigid_surf_points, self._rigid_surf_normals,
                self._rigid_surf_restitutions, self._rigid_surf_frictions,
                self._rigid_surf_start_times, self._rigid_surf_end_times,
                len(self.rigid_surface_colliders), self.time, dt,
            ],
            device=device,
        )

    def sync_rigid_particle_state(self, device="cuda:0"):
        """Push current rigid body state (x_cm, v_cm, omega, orientation) into
        per-particle arrays.  Call this once after finalize_rigid_bodies() and
        any set_rigid_body_velocity() calls, before the first p2g2p() step."""
        if self.n_rigid_bodies > 0:
            wp.launch(kernel=rigid_particle_update, dim=self.n_particles,
                      inputs=[self.mpm_state, self.mpm_model, self.rigid_x_cm, self.rigid_v_cm,
                               self.rigid_omega, self.rigid_orientation],
                      device=device)

    def p2g2p(self, step, dt, device="cuda:0"):
        grid_size = (
            self.n_envs,
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )
        wp.launch(
            kernel=zero_grid,
            dim=(grid_size),
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )

        # apply pre-p2g operations on particles
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            )
        # apply dirichlet particle v modifier
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel = self.particle_velocity_modifiers[k],
                dim = self.n_particles,
                inputs=[self.time, self.mpm_state, self.particle_velocity_modifier_params[k]],
                device=device,
            )

        # apply per-env external body force (RL actions) to particle velocities
        wp.launch(
            kernel=apply_external_force_per_env,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model, dt],
            device=device,
        )

        # compute stress = stress(returnMap(F_trial))
        with wp.ScopedTimer(
            "compute_stress_from_F_trial",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # F and stress are updated

        # p2g
        with wp.ScopedTimer(
            "p2g",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=p2g_apic_with_stress,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # apply p2g'

        # grid update
        with wp.ScopedTimer(
            "grid_update", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )

        if self.mpm_model.grid_v_damping_scale < 1.0:
            wp.launch(
                kernel=add_damping_via_grid,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model.grid_v_damping_scale],
                device=device,
            )

        # apply BC on grid
        with wp.ScopedTimer(
            "apply_BC_on_grid", synchronize=True, print=False, dict=self.time_profile
        ):
            for k in range(len(self.grid_postprocess)):
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,
                    inputs=[
                        self.time,
                        dt,
                        self.mpm_state,
                        self.mpm_model,
                        self.collider_params[k],
                    ],
                    device=device,
                )
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])

        # g2p
        with wp.ScopedTimer(
            "g2p", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=g2p,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # x, v, C, F_trial are updated

        # rigid body step (skipped when no rigid bodies are present)
        if self.n_rigid_bodies > 0:
            with wp.ScopedTimer("rigid_body", synchronize=True, print=False, dict=self.time_profile):
                # zero accumulation buffers
                wp.launch(kernel=set_vec3_to_zero, dim=self.n_rigid_bodies,
                          inputs=[self._rigid_linear_mom], device=device)
                wp.launch(kernel=set_vec3_to_zero, dim=self.n_rigid_bodies,
                          inputs=[self._rigid_angular_mom], device=device)
                # gather grid momentum weighted by particle mass
                wp.launch(kernel=rigid_g2p_accumulate, dim=self.n_particles,
                          inputs=[self.mpm_state, self.mpm_model,
                                  self.rigid_x_cm, self._rigid_linear_mom,
                                  self._rigid_angular_mom],
                          device=device)
                # integrate rigid body EOM and update orientation
                wp.launch(kernel=rigid_body_integrate, dim=self.n_rigid_bodies,
                          inputs=[self.rigid_x_cm, self.rigid_v_cm, self.rigid_omega,
                                  self.rigid_orientation, self.rigid_mass,
                                  self.rigid_inv_inertia_body,
                                  self._rigid_linear_mom, self._rigid_angular_mom, dt],
                          device=device)
                # apply restitution impulses to rigid bodies (surface colliders with e > 0)
                self._apply_rigid_restitution(dt, device=device)
                # push updated state back to particles
                wp.launch(kernel=rigid_particle_update, dim=self.n_particles,
                          inputs=[self.mpm_state, self.mpm_model, self.rigid_x_cm, self.rigid_v_cm,
                                  self.rigid_omega, self.rigid_orientation],
                          device=device)

        #### CFL check ####
        # particle_v = self.mpm_state.particle_v.numpy()
        # if np.max(np.abs(particle_v)) > self.mpm_model.dx / dt:
        #     print("max particle v: ", np.max(np.abs(particle_v)))
        #     print("max allowed  v: ", self.mpm_model.dx / dt)
        #     print("does not allow v*dt>dx")
        #     input()
        #### CFL check ####
        self.time = self.time + dt

    # set particle densities to all_particle_densities, 
    def reset_densities_and_update_masses(self, all_particle_densities, device = "cuda:0"):
        all_particle_densities = all_particle_densities.clone().detach()
        self.mpm_state.particle_density = torch2warp_float(all_particle_densities, dvc=device)
        wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )

    # clone = True makes a copy, not necessarily needed
    def import_particle_x_from_torch(self, tensor_x, clone=True, device="cuda:0"):
        if tensor_x is not None:
            if clone:
                tensor_x = tensor_x.clone().detach()
            self.mpm_state.particle_x = torch2warp_vec3(tensor_x, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_v_from_torch(self, tensor_v, clone=True, device="cuda:0"):
        if tensor_v is not None:
            if clone:
                tensor_v = tensor_v.clone().detach()
            self.mpm_state.particle_v = torch2warp_vec3(tensor_v, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_F_from_torch(self, tensor_F, clone=True, device="cuda:0"):
        if tensor_F is not None:
            if clone:
                tensor_F = tensor_F.clone().detach()
            tensor_F = torch.reshape(tensor_F, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_F = torch2warp_mat33(tensor_F, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_C_from_torch(self, tensor_C, clone=True, device="cuda:0"):
        if tensor_C is not None:
            if clone:
                tensor_C = tensor_C.clone().detach()
            tensor_C = torch.reshape(tensor_C, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_C = torch2warp_mat33(tensor_C, dvc=device)
            
    def import_particle_selection_from_torch(self, tensor_selection, clone=True, device="cuda:0"):
        if tensor_selection is not None:
            if clone:
                tensor_selection = tensor_selection.clone().detach()
            self.mpm_state.particle_selection = torch2warp_int(tensor_selection, dvc=device)

    def import_particle_material_from_torch(self, tensor_material, clone=True, device="cuda:0"):
        if tensor_material is not None:
            if clone:
                tensor_material = tensor_material.clone().detach()
            self.mpm_state.particle_material = torch2warp_int(tensor_material, dvc=device)

    def export_particle_material_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_material)

    def set_parameters_for_particles(self, start_idx, end_idx, params_dict, device="cuda:0"):
        """Set material type and parameters for particles in range [start_idx, end_idx)."""
        import torch

        mat_id = None
        if "material" in params_dict:
            mat_id = self._material_name_to_id(params_dict["material"])
            # set particle_material for the range
            mat_torch = wp.to_torch(self.mpm_state.particle_material)
            mat_torch[start_idx:end_idx] = mat_id

        if mat_id is None:
            mat_id = self.mpm_model.material

        # per-type parameters (write to model arrays at mat_id index)
        if "E" in params_dict:
            E_torch = wp.to_torch(self.mpm_model.E)
            E_torch[mat_id] = params_dict["E"]
        if "nu" in params_dict:
            nu_torch = wp.to_torch(self.mpm_model.nu)
            nu_torch[mat_id] = params_dict["nu"]
        if "bulk_modulus" in params_dict:
            bulk_torch = wp.to_torch(self.mpm_model.bulk)
            bulk_torch[mat_id] = params_dict["bulk_modulus"]
        if "hardening" in params_dict:
            h_torch = wp.to_torch(self.mpm_model.hardening)
            h_torch[mat_id] = float(params_dict["hardening"])
        if "xi" in params_dict:
            xi_torch = wp.to_torch(self.mpm_model.xi)
            xi_torch[mat_id] = params_dict["xi"]
        if "friction_angle" in params_dict:
            sin_phi = wp.sin(params_dict["friction_angle"] / 180.0 * 3.14159265)
            alpha_val = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
            alpha_torch = wp.to_torch(self.mpm_model.alpha)
            alpha_torch[mat_id] = alpha_val
        if "plastic_viscosity" in params_dict:
            pv_torch = wp.to_torch(self.mpm_model.plastic_viscosity)
            pv_torch[mat_id] = params_dict["plastic_viscosity"]
        if "softening" in params_dict:
            s_torch = wp.to_torch(self.mpm_model.softening)
            s_torch[mat_id] = params_dict["softening"]

        # rigid body id — only meaningful when material == "rigid" (8)
        if "obj_id" in params_dict:
            rid_torch = wp.to_torch(self.mpm_state.particle_rigid_id)
            rid_torch[start_idx:end_idx] = int(params_dict["obj_id"])

        # per-particle parameters (set for the range only)
        if "density" in params_dict:
            density_torch = wp.to_torch(self.mpm_state.particle_density)
            density_torch[start_idx:end_idx] = params_dict["density"]
            mass_torch = wp.to_torch(self.mpm_state.particle_mass)
            vol_torch = wp.to_torch(self.mpm_state.particle_vol)
            mass_torch[start_idx:end_idx] = density_torch[start_idx:end_idx] * vol_torch[start_idx:end_idx]

        if "yield_stress" in params_dict:
            ys_torch = wp.to_torch(self.mpm_model.yield_stress)
            ys_torch[start_idx:end_idx] = params_dict["yield_stress"]

        # compute mu/lam for this range from per-type E/nu
        if "E" in params_dict or "nu" in params_dict:
            E_torch = wp.to_torch(self.mpm_model.E)
            nu_torch = wp.to_torch(self.mpm_model.nu)
            E_val = float(E_torch[mat_id])
            nu_val = float(nu_torch[mat_id])
            mu_val = E_val / (2.0 * (1.0 + nu_val))
            lam_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
            mu_torch = wp.to_torch(self.mpm_model.mu)
            lam_torch = wp.to_torch(self.mpm_model.lam)
            mu_torch[start_idx:end_idx] = mu_val
            lam_torch[start_idx:end_idx] = lam_val

    def export_particle_x_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_x)

    def export_particle_v_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_v)
    
    def export_particle_vol_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_vol)
    
    def export_particle_stress_to_torch(self):
        stress_tensor = wp.to_torch(self.mpm_state.particle_stress)
        stress_tensor = stress_tensor.reshape(-1, 9)
        return stress_tensor

    def export_particle_F_to_torch(self):
        F_tensor = wp.to_torch(self.mpm_state.particle_F)
        F_tensor = F_tensor.reshape(-1, 9)
        return F_tensor

    def export_particle_R_to_torch(self, device="cuda:0"):
        with wp.ScopedTimer(
            "compute_R_from_F",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_R_from_F,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )

        R_tensor = wp.to_torch(self.mpm_state.particle_R)
        R_tensor = R_tensor.reshape(-1, 9)
        return R_tensor

    def export_particle_C_to_torch(self):
        C_tensor = wp.to_torch(self.mpm_state.particle_C)
        C_tensor = C_tensor.reshape(-1, 9)
        return C_tensor

    def export_particle_cov_to_torch(self, device="cuda:0"):
        if not self.mpm_model.update_cov_with_F:
            with wp.ScopedTimer(
                "compute_cov_from_F",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=compute_cov_from_F,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model],
                    device=device,
                )

        cov = wp.to_torch(self.mpm_state.particle_cov)
        return cov
    def export_particle_selection_to_torch(self):
        selection_tensor = wp.to_torch(self.mpm_state.particle_selection)
        return selection_tensor
    def export_grid_v_out_to_torch(self):
        grid_v_tensor = wp.to_torch(self.mpm_state.grid_v_out)
        return grid_v_tensor
    def export_grid_v_in_to_torch(self):
        grid_v_tensor = wp.to_torch(self.mpm_state.grid_v_in)
        return grid_v_tensor
    def export_grid_m_to_torch(self):
        grid_m_tensor = wp.to_torch(self.mpm_state.grid_m)
        return grid_m_tensor
    
    def print_time_profile(self):
        print("MPM Time profile:")
        for key, value in self.time_profile.items():
            print(key, sum(value))

    # a surface specified by a point and the normal vector
    def add_surface_collider(
        self,
        point,
        normal,
        surface="sticky",
        friction=0.0,
        restitution=0.0,
        start_time=0.0,
        end_time=999.0,
    ):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(float(sum(x**2 for x in normal)))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])

        if surface == "sticky" and friction != 0:
            raise ValueError("friction must be 0 on sticky surfaces.")
        if surface == "sticky":
            collider_param.surface_type = 0
        elif surface == "slip":
            collider_param.surface_type = 1
        elif surface == "cut":
            collider_param.surface_type = 11
        else:
            collider_param.surface_type = 2
        # frictional
        collider_param.friction = friction

        if restitution != 0.0:
            if not (0.0 < restitution <= 1.0):
                raise ValueError("restitution must be between 0 (exclusive) and 1 (inclusive).")
            self.rigid_surface_colliders.append({
                "point": list(point),
                "normal": list(normal),
                "friction": friction,
                "restitution": restitution,
                "start_time": start_time,
                "end_time": end_time,
            })
            self._rigid_surf_uploaded = False  # will be re-uploaded before first use

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            e, grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)

                if dotproduct < 0.0:
                    if param.surface_type == 0:
                        state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )
                    elif param.surface_type == 11:
                        if (
                            float(grid_z) * model.dx < 0.4
                            or float(grid_z) * model.dx > 0.53
                        ):
                            state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(
                                0.0, 0.0, 0.0
                            )
                        else:
                            v_in = state.grid_v_out[e, grid_x, grid_y, grid_z]
                            state.grid_v_out[e, grid_x, grid_y, grid_z] = (
                                wp.vec3(v_in[0], 0.0, v_in[2]) * 0.3
                            )
                    else:
                        v = state.grid_v_out[e, grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        if param.surface_type == 1:
                            v = (
                                v - normal_component * n
                            )  # Project out all normal component
                        else:
                            v = (
                                v - wp.min(normal_component, 0.0) * n
                            )  # Project out only inward normal component
                        if normal_component < 0.0 and wp.length(v) > 1e-20:
                            v = wp.max(
                                0.0, wp.length(v) + normal_component * param.friction
                            ) * wp.normalize(
                                v
                            )  # apply friction here
                        state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(
                            v[0], v[1], v[2]
                        )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # a cubiod is a rectangular cube'
    # centered at `point`
    # dimension is x: point[0]±size[0]
    #              y: point[1]±size[1]
    #              z: point[2]±size[2]
    # all grid nodes lie within the cubiod will have their speed set to velocity
    # the cuboid itself is also moving with const speed = velocity
    # set the speed to zero to fix BC
    def set_velocity_on_cuboid(
        self,
        point,
        size,
        velocity,
        start_time=0.0,
        end_time=999.0,
        reset=0,
    ):
        point = list(point)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.size = size
        collider_param.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
        # collider_param.threshold = threshold
        collider_param.reset = reset
        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            e, grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                if (
                    wp.abs(offset[0]) < param.size[0]
                    and wp.abs(offset[1]) < param.size[1]
                    and wp.abs(offset[2]) < param.size[2]
                ):
                    state.grid_v_out[e, grid_x, grid_y, grid_z] = param.velocity
            elif param.reset == 1:
                if time < param.end_time + 15.0 * dt:
                    state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        def modify(time, dt, param: Dirichlet_collider):
            if time >= param.start_time and time < param.end_time:
                param.point = wp.vec3(
                    param.point[0] + dt * param.velocity[0],
                    param.point[1] + dt * param.velocity[1],
                    param.point[2] + dt * param.velocity[2],
                )  # param.point + dt * param.velocity

        self.grid_postprocess.append(collide)
        self.modify_bc.append(modify)

    def add_bounding_box(self, start_time=0.0, end_time=999.0):
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            e, grid_x, grid_y, grid_z = wp.tid()
            padding = 3
            if time >= param.start_time and time < param.end_time:
                if grid_x < padding and state.grid_v_out[e, grid_x, grid_y, grid_z][0] < 0:
                    state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v_out[e, grid_x, grid_y, grid_z][1],
                        state.grid_v_out[e, grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_x >= model.grid_dim_x - padding
                    and state.grid_v_out[e, grid_x, grid_y, grid_z][0] > 0
                ):
                    state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v_out[e, grid_x, grid_y, grid_z][1],
                        state.grid_v_out[e, grid_x, grid_y, grid_z][2],
                    )

                if grid_y < padding and state.grid_v_out[e, grid_x, grid_y, grid_z][1] < 0:
                    state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[e, grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v_out[e, grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_y >= model.grid_dim_y - padding
                    and state.grid_v_out[e, grid_x, grid_y, grid_z][1] > 0
                ):
                    state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[e, grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v_out[e, grid_x, grid_y, grid_z][2],
                    )

                if grid_z < padding and state.grid_v_out[e, grid_x, grid_y, grid_z][2] < 0:
                    state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[e, grid_x, grid_y, grid_z][0],
                        state.grid_v_out[e, grid_x, grid_y, grid_z][1],
                        0.0,
                    )
                if (
                    grid_z >= model.grid_dim_z - padding
                    and state.grid_v_out[e, grid_x, grid_y, grid_z][2] > 0
                ):
                    state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[e, grid_x, grid_y, grid_z][0],
                        state.grid_v_out[e, grid_x, grid_y, grid_z][1],
                        0.0,
                    )

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    # particle_v += force/particle_mass * dt
    # this is applied from start_dt, ends after num_dt p2g2p's
    # particle velocity is changed before p2g at each timestep
    def add_impulse_on_particles(self, force, dt, point =[1,1,1], size = [1,1,1], num_dt = 1, start_time=0.0, device = "cuda:0"):
        impulse_param = Impulse_modifier()
        impulse_param.start_time = start_time
        impulse_param.end_time = start_time + dt * num_dt

        impulse_param.point = wp.vec3(point[0], point[1], point[2])
        impulse_param.size = wp.vec3(size[0], size[1], size[2])
        impulse_param.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device)

        impulse_param.force = wp.vec3(
            force[0],
            force[1],
            force[2],
        )

        wp.launch(
                kernel=selection_add_impulse_on_particles,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state,
                    impulse_param
                ],
                device=device,
            )

        self.impulse_params.append(impulse_param)

        @wp.kernel
        def apply_force(
            time: float, dt: float, state: MPMStateStruct, param: Impulse_modifier
        ):
            p = wp.tid()
            if time >= param.start_time and time < param.end_time:
                if param.mask[p] == 1:
                    impulse = wp.vec3(
                        param.force[0] / state.particle_mass[p],
                        param.force[1] / state.particle_mass[p],
                        param.force[2] / state.particle_mass[p],
                    )
                    state.particle_v[p] = state.particle_v[p] + impulse * dt

        self.pre_p2g_operations.append(apply_force)



    def enforce_particle_velocity_translation(self, point, size, velocity, start_time, end_time, device = "cuda:0"):

        # first select certain particles based on position

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.size = wp.vec3(size[0], size[1], size[2])

        velocity_modifier_params.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device)
        
        wp.launch(
                kernel=selection_enforce_particle_velocity_translation,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state,
                    velocity_modifier_params
                ],
                device=device,
            )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)
        
        @wp.kernel
        def modify_particle_v_before_p2g(time: float,
                                        state: MPMStateStruct,
                                        velocity_modifier_params: ParticleVelocityModifier):
            p = wp.tid()
            if time >= velocity_modifier_params.start_time and time < velocity_modifier_params.end_time:
                if velocity_modifier_params.mask[p] == 1:
                    state.particle_v[p] = velocity_modifier_params.velocity

        
        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)


    # define a cylinder with center point, half_height, radius, normal
    # particles within the cylinder are rotating along the normal direction
    # may also have a translational velocity along the normal direction
    def enforce_particle_velocity_rotation(self, point, normal, 
                        half_height_and_radius, rotation_scale, translation_scale, start_time, end_time, device = "cuda:0"):

        normal_scale = 1.0 / wp.sqrt(float(normal[0]**2 + normal[1]**2 + normal[2]**2))
        normal = list(normal_scale * x for x in normal)

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.half_height_and_radius = wp.vec2(half_height_and_radius[0], half_height_and_radius[1])
        velocity_modifier_params.normal = wp.vec3(normal[0], normal[1], normal[2])

        horizontal_1 = wp.vec3(1.0,1.0,1.0)
        if wp.abs(wp.dot(velocity_modifier_params.normal, horizontal_1)) < 0.01:
            horizontal_1 = wp.vec3(0.72, 0.37, -0.67)
        horizontal_1 = horizontal_1 - wp.dot(horizontal_1, velocity_modifier_params.normal) * velocity_modifier_params.normal
        horizontal_1 = horizontal_1 * (1.0 / wp.length(horizontal_1))
        horizontal_2 = wp.cross(horizontal_1, velocity_modifier_params.normal)

        velocity_modifier_params.horizontal_axis_1 = horizontal_1
        velocity_modifier_params.horizontal_axis_2 = horizontal_2

        velocity_modifier_params.rotation_scale = rotation_scale
        velocity_modifier_params.translation_scale = translation_scale

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device)
        
        wp.launch(
                kernel=selection_enforce_particle_velocity_cylinder,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state,
                    velocity_modifier_params
                ],
                device=device,
            )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)
        
        @wp.kernel
        def modify_particle_v_before_p2g(time: float,
                                        state: MPMStateStruct,
                                        velocity_modifier_params: ParticleVelocityModifier):
            p = wp.tid()
            if time >= velocity_modifier_params.start_time and time < velocity_modifier_params.end_time:
                if velocity_modifier_params.mask[p] == 1:
                    offset = state.particle_x[p] - velocity_modifier_params.point
                    horizontal_distance = wp.length(offset - wp.dot(offset, velocity_modifier_params.normal) * velocity_modifier_params.normal)
                    cosine = wp.dot(offset, velocity_modifier_params.horizontal_axis_1) / horizontal_distance
                    theta = wp.acos(cosine)
                    if wp.dot(offset, velocity_modifier_params.horizontal_axis_2) > 0:
                        theta = theta
                    else:
                        theta = -theta
                    axis1_scale = - horizontal_distance * wp.sin(theta) * velocity_modifier_params.rotation_scale
                    axis2_scale = horizontal_distance * wp.cos(theta) * velocity_modifier_params.rotation_scale
                    axis_vertical_scale = translation_scale
                    state.particle_v[p] = axis1_scale * velocity_modifier_params.horizontal_axis_1 + axis2_scale * velocity_modifier_params.horizontal_axis_2 + axis_vertical_scale * velocity_modifier_params.normal 
                        

        
        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)
        

    # Add a point cloud as a static collider.
    # point_cloud_np: numpy array of shape (N, 3) with positions in world coordinates.
    # padding: number of grid cells to dilate around each occupied cell (fills gaps in sparse clouds).
    def add_point_cloud_collider(self, point_cloud_np, padding=1, start_time=0.0, end_time=999.0, device="cuda:0"):
        import numpy as np

        n_grid = self.mpm_model.n_grid
        inv_dx = self.mpm_model.inv_dx

        # Convert point positions to grid indices
        indices = (point_cloud_np * inv_dx).astype(np.int32)
        indices = np.clip(indices, 0, n_grid - 1)

        # Build occupancy grid
        occupancy = np.zeros((n_grid, n_grid, n_grid), dtype=np.int32)
        occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

        # Dilate occupancy to fill gaps (without wrap-around)
        if padding > 0:
            for _ in range(padding):
                dilated = occupancy.copy()
                for axis in range(3):
                    slices_src = [slice(None)] * 3
                    slices_dst = [slice(None)] * 3
                    slices_src[axis] = slice(0, -1)
                    slices_dst[axis] = slice(1, None)
                    dilated[tuple(slices_dst)] |= occupancy[tuple(slices_src)]
                    slices_src = [slice(None)] * 3
                    slices_dst = [slice(None)] * 3
                    slices_src[axis] = slice(1, None)
                    slices_dst[axis] = slice(0, -1)
                    dilated[tuple(slices_dst)] |= occupancy[tuple(slices_src)]
                occupancy = dilated

        num_occupied = int(occupancy.sum())
        print(f"Point cloud collider: {len(point_cloud_np)} points -> {num_occupied} occupied grid cells (padding={padding})")

        collider_param = PointCloudCollider()
        collider_param.occupancy_grid = wp.from_numpy(occupancy, dtype=int, device=device)
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: PointCloudCollider,
        ):
            e, grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                if param.occupancy_grid[grid_x, grid_y, grid_z] == 1:
                    state.grid_v_out[e, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    def add_sdf_collider(
        self,
        sdf_np,
        origin,
        dx_sdf,
        surface="slip",
        friction=0.0,
        start_time=0.0,
        end_time=999.0,
        initial_translation=None,
        initial_rotation=None,
        device="cuda:0",
    ):
        """Add an SDF-based surface collider.

        Parameters
        ----------
        sdf_np : numpy.ndarray, shape (nx, ny, nz), dtype float32
            Signed distance field in the collider's local frame.
            φ < 0 inside the object, φ > 0 outside.
            Typical construction from an occupancy grid::

                from scipy.ndimage import distance_transform_edt
                d_out = distance_transform_edt(1 - occupancy) * dx_sdf
                d_in  = distance_transform_edt(occupancy)     * dx_sdf
                sdf_np = (d_out - d_in).astype(np.float32)

        origin : array-like, length 3
            World-space (or local-frame) position of sdf_np[0, 0, 0].
        dx_sdf : float
            Voxel size of the SDF grid.  Should be smaller than the MPM
            grid spacing (``self.mpm_model.dx``) for sub-cell accuracy.
        surface : str
            ``'sticky'``, ``'slip'``, or ``'frictional'``.
        friction : float
            Coulomb friction coefficient (only used when surface='frictional').
        start_time / end_time : float
            Time window during which this collider is active.
        initial_translation : array-like, length 3, optional
            Initial world-space position of the object origin.
            Defaults to (0, 0, 0) — use this when the SDF is already
            expressed in world space.
        initial_rotation : array-like, shape (3, 3), optional
            Initial rotation matrix (object → world).
            Defaults to the identity matrix.
        device : str

        Returns
        -------
        param : SDFCollider
            The collider struct stored on the GPU.  Pass this to a
            ``modify_bc`` callback to update ``translation`` and
            ``rotation`` each substep for moving objects::

                def move(time, dt, param):
                    t, R = your_trajectory(time)
                    param.translation = wp.vec3(*t)
                    param.rotation    = wp.mat33(*R.flatten())
        """
        sdf_np = np.ascontiguousarray(sdf_np, dtype=np.float32)
        nx, ny, nz = sdf_np.shape

        if initial_translation is None:
            initial_translation = [0.0, 0.0, 0.0]
        if initial_rotation is None:
            initial_rotation = np.eye(3, dtype=np.float32)
        initial_rotation = np.asarray(initial_rotation, dtype=np.float32)

        surface_map = {"sticky": 0, "slip": 1, "frictional": 2}
        if surface not in surface_map:
            raise ValueError(f"surface must be 'sticky', 'slip', or 'frictional', got '{surface}'")

        param = SDFCollider()
        param.sdf          = wp.from_numpy(sdf_np, dtype=float, device=device)
        param.origin       = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        param.dx_sdf       = float(dx_sdf)
        param.nx           = nx
        param.ny           = ny
        param.nz           = nz
        param.surface_type = surface_map[surface]
        param.friction     = float(friction)
        param.start_time   = float(start_time)
        param.end_time     = float(end_time)
        param.translation  = wp.vec3(float(initial_translation[0]),
                                     float(initial_translation[1]),
                                     float(initial_translation[2]))
        R = initial_rotation
        param.rotation = wp.mat33(
            float(R[0, 0]), float(R[0, 1]), float(R[0, 2]),
            float(R[1, 0]), float(R[1, 1]), float(R[1, 2]),
            float(R[2, 0]), float(R[2, 1]), float(R[2, 2]),
        )

        self.collider_params.append(param)
        self.grid_postprocess.append(collide_sdf)
        self.modify_bc.append(None)
        return param

    # given normal direction, say [0,0,1]
    # gradually release grid velocities from start position to end position
    def release_particles_sequentially(self, normal, start_position, end_position, num_layers, start_time, end_time):
        num_layers = 50
        point = [0,0,0]
        size = [0,0,0]
        axis = -1
        for i in range(3):
            if normal[i] == 0:
                point[i] = 1
                size[i] = 1
            else:
                axis = i
                point[i] = end_position
            
        half_length_portion = wp.abs(start_position - end_position)/num_layers
        end_time_portion = end_time / num_layers
        for i in range(num_layers):
            size[axis] = half_length_portion * (num_layers - i)
            self.enforce_particle_velocity_translation(point=point, size=size, velocity = [0,0,0], start_time=start_time, end_time=end_time_portion * (i+1))




        


