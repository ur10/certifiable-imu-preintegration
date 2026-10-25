import numpy as np
import scipy.sparse as sp
import gtsam
from dataclasses import dataclass
from gtsam.utils.plot import plot_pose3
from matplotlib import pyplot as plt
from gtsam.symbol_shorthand import B, V, X
from mosek.fusion import Model, Domain, Expr, Matrix, ObjectiveSense
import argparse


def skew(v):
	"""Return the skew-symmetric matrix of a 3D vector."""
	v = np.asarray(v).reshape(3)
	return np.array([
		[0.0, -v[2], v[1]],
		[v[2], 0.0, -v[0]],
		[-v[1], v[0], 0.0],
	])


def so3_exp(phi):
	"""Compute the SO(3) exponential map using Rodrigues' formula."""
	phi = np.asarray(phi).reshape(3)
	angle = np.linalg.norm(phi)
	if angle < 1e-6:
		return np.eye(3) + skew(phi)

	K = skew(phi / angle)
	return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def right_jacobian_so3(phi):
	"""Compute the right Jacobian of SO(3)."""
	phi = np.asarray(phi).reshape(3)
	angle = np.linalg.norm(phi)
	if angle < 1e-6:
		return np.eye(3) - 0.5 * skew(phi)

	phi_hat = skew(phi)
	term1 = (1.0 - np.cos(angle)) / (angle ** 2)
	term2 = (angle - np.sin(angle)) / (angle ** 3)
	return np.eye(3) - term1 * phi_hat + term2 * (phi_hat @ phi_hat)


def propagate_rotation_jacobian(imu_samples, gyro_bias):
	"""Propagate preintegrated rotation and its Jacobian w.r.t. gyro bias.

	Args:
		imu_samples: sequence of samples with `timestamp_ns` and `gyro_rad_s`.
		gyro_bias: 3D gyro bias vector.

	Returns:
		(delta_r, dR_dbg): the preintegrated rotation and analytical Jacobian.
	"""
	if len(imu_samples) < 2:
		return np.eye(3), np.zeros((3, 3))

	gyro_bias = np.asarray(gyro_bias).reshape(3)
	delta_r = np.eye(3)
	dR_dbg = np.zeros((3, 3))

	for sample_k, sample_k1 in zip(imu_samples[:-1], imu_samples[1:]):
		dt = (sample_k1.timestamp_ns - sample_k.timestamp_ns) * 1e-9
		omega_k = np.asarray(sample_k.gyro_rad_s).reshape(3) - gyro_bias
		omega_k1 = np.asarray(sample_k1.gyro_rad_s).reshape(3) - gyro_bias
		omega_mid = 0.5 * (omega_k + omega_k1)
		dtheta = omega_mid * dt

		dR = so3_exp(dtheta)
		jr = right_jacobian_so3(dtheta)
		dR_dbg = dR.T @ dR_dbg - jr * dt
		delta_r = delta_r @ dR

	return delta_r, dR_dbg


def build_scenarios(velocity=None):
	"""Return the default scenarios dict and a default velocity vector."""
	if velocity is None:
		velocity = np.array([2, 0, 0])

	scenarios = {
		"zero_twist": (np.zeros(3), np.zeros(3)),
		"forward_motion": (np.zeros(3), velocity),
		"loop": (np.array([0, -np.radians(30), 0]), velocity),
		"sick": (np.array([np.radians(30), -np.radians(30), 0]), velocity),
	}
	return scenarios


def plot_scenario(scenario, T, dt, title="IMU trajectory scenario", fignum=0, maxDim=5):
	"""Plot poses from a gtsam scenario over time.

	Args:
		scenario: a gtsam.ConstantTwistScenario-like object with .pose(t).
		T: total time to simulate.
		dt: timestep.
		title: figure title.
		fignum: matplotlib figure number.
		maxDim: initial axis limit.
	"""
	for t in np.arange(0, T, dt):
		actualPose = scenario.pose(t)
		plot_pose3(fignum, actualPose, axis_length=0.3)

		translation = actualPose.translation()
		maxDim = max([max(np.abs(translation)), maxDim])
		ax = plt.gca()
		ax.set_xlim3d(-maxDim, maxDim)
		ax.set_ylim3d(-maxDim, maxDim)
		ax.set_zlim3d(-maxDim, maxDim)
		ax.set_title(title)

	plt.show()


def create_preintegration(acc_bias=None, gyro_bias=None, gravity=9.81, gyro_sigma=1e-3, accel_sigma=1e-3, integration_sigma=1e-7):
	"""Create PreintegrationParams, actual bias, and a PreintegratedImuMeasurements object.

	Returns: (actualBias, pim_params, pim)
	"""
	if acc_bias is None:
		acc_bias = np.array([-0.3, 0.1, 0.2])
	if gyro_bias is None:
		gyro_bias = np.array([0.1, 0.3, -0.1])

	actualBias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)

	pim_params = gtsam.PreintegrationParams.MakeSharedU(gravity)
	I_3x3 = np.eye(3)
	pim_params.setGyroscopeCovariance((gyro_sigma or 0.0) ** 2 * I_3x3)
	pim_params.setAccelerometerCovariance((accel_sigma or 0.0) ** 2 * I_3x3)
	pim_params.setIntegrationCovariance((integration_sigma or 0.0) ** 2 * I_3x3)

	pim = gtsam.PreintegratedImuMeasurements(pim_params, actualBias)
	return actualBias, pim_params, pim


def _run_preintegration_sequence(pim_params, bias, acc_measurements, gyro_measurements, dt):
	"""Rebuild a PreintegratedImuMeasurements object from raw IMU measurements."""
	pim = gtsam.PreintegratedImuMeasurements(pim_params, bias)
	for acc, gyro in zip(acc_measurements, gyro_measurements):
		pim.integrateMeasurement(np.asarray(acc), np.asarray(gyro), dt)
	return pim


def extract_preintegration_data(pim,
						 compute_bias_jacobians=False,
						 pim_params=None,
						 bias=None,
						 acc_measurements=None,
						 gyro_measurements=None,
						 imu_samples=None,
						 dt=None,
						 eps=1e-6):
	"""Return preintegration deltas and optional bias Jacobians.

	Args:
		pim: PreintegratedImuMeasurements object already integrated.
		compute_bias_jacobians: if True, compute numeric Jacobians w.r.t. bias.
		pim_params: PreintegrationParams used for integration.
		bias: imuBias.ConstantBias used for the integration.
		acc_measurements: list/array of accelerometer measurements for jacobian recomputation.
		gyro_measurements: list/array of gyroscope measurements for jacobian recomputation.
		imu_samples: optional sequence with timestamped IMU samples for analytical rotation Jacobian.
		dt: timestep for the measurement sequence.
		eps: finite-difference step size.

	Returns:
		A dict containing deltaPij, deltaVij, deltaRij, deltaTij, preintMeasCov, preintegrated.
		If compute_bias_jacobians=True, also includes jacobians for acc and gyro bias.
	"""
	data = {
		"deltaRij": pim.deltaRij(),
		"deltaVij": pim.deltaVij(),
		"deltaPij": pim.deltaPij(),
		"deltaTij": pim.deltaTij(),
		"preintMeasCov": np.array(pim.preintMeasCov()),
		"preintegrated": np.array(pim.preintegrated()),
	}

	if not compute_bias_jacobians:
		return data

	if pim_params is None or bias is None or acc_measurements is None or gyro_measurements is None or dt is None:
		raise ValueError(
			"To compute bias Jacobians, provide pim_params, bias, acc_measurements, gyro_measurements, and dt.")

	# Analytical rotation Jacobian from the gyro measurements, when timestamps are available.
	# This keeps the rotation-bias propagation separate from the finite-difference Jacobians
	# used for the full preintegration state.
	data["rotation_jacobian"] = None
	if imu_samples is not None and len(imu_samples) > 0:
		_, dR_dbg = propagate_rotation_jacobian(imu_samples, bias.gyroscope())
		data["rotation_jacobian"] = dR_dbg

	jacobians = {
		"acc_bias": {
			"deltaPij": np.zeros((3, 3)),
			"deltaVij": np.zeros((3, 3)),
			"deltaRij": np.zeros((3, 3)),
			"deltaTij": np.zeros((1, 3)),
		},
		"gyro_bias": {
			"deltaPij": np.zeros((3, 3)),
			"deltaVij": np.zeros((3, 3)),
			"deltaRij": np.zeros((3, 3)),
			"deltaTij": np.zeros((1, 3)),
		},
	}

	base_R = np.asarray(gtsam.Rot3.Logmap(pim.deltaRij()))
	base_V = np.asarray(pim.deltaVij())
	base_P = np.asarray(pim.deltaPij())
	base_T = np.asarray(pim.deltaTij())

	for bias_type, base_bias in [("acc_bias", bias.accelerometer()), ("gyro_bias", bias.gyroscope())]:
		for j in range(3):
			perturbed = np.array(base_bias)
			perturbed[j] += eps
			if bias_type == "acc_bias":
				perturbed_bias = gtsam.imuBias.ConstantBias(perturbed, np.array(bias.gyroscope()))
			else:
				perturbed_bias = gtsam.imuBias.ConstantBias(np.array(bias.accelerometer()), perturbed)

			pim_perturbed = _run_preintegration_sequence(pim_params, perturbed_bias, acc_measurements, gyro_measurements, dt)
			jac_R = (np.asarray(gtsam.Rot3.Logmap(pim_perturbed.deltaRij())) - base_R) / eps
			jac_V = (np.asarray(pim_perturbed.deltaVij()) - base_V) / eps
			jac_P = (np.asarray(pim_perturbed.deltaPij()) - base_P) / eps
			jac_T = (np.asarray(pim_perturbed.deltaTij()) - base_T) / eps

			jacobians[bias_type]["deltaRij"][:, j] = jac_R
			jacobians[bias_type]["deltaVij"][:, j] = jac_V
			jacobians[bias_type]["deltaPij"][:, j] = jac_P
			jacobians[bias_type]["deltaTij"][0, j] = jac_T

	data["bias_jacobians"] = jacobians
	return data


def main_loop(runner, scenario, pim, graph, initial, T, dt, return_cov=False,
							 init_rot_noise=0.1, init_trans_noise=1.0, init_vel_noise=0.1):
	"""Run the main preintegration loop, add ImuFactor every second.

	Args:
		runner: gtsam.ScenarioRunner
		scenario: gtsam scenario providing navState(t)
		pim: PreintegratedImuMeasurements to integrate into
		graph: gtsam.NonlinearFactorGraph
		initial: gtsam.Values for initial estimates
		T: total time
		dt: timestep
		return_cov: if True, also return covariance history and times
		init_rot_noise: standard deviation for initial pose rotation noise
		init_trans_noise: standard deviation for initial pose translation noise
		init_vel_noise: standard deviation for initial velocity noise

	Returns:
		(graph, initial) or (graph, initial, cov_history, times)
	"""
	i = 0
	cov_history = []
	times = []

	keyframe_covs = []
	keyframe_times = []

	for k, t in enumerate(np.arange(0, T, dt)):
		measuredOmega = runner.measuredAngularVelocity(t)
		measuredAcc = runner.measuredSpecificForce(t)

		pim.integrateMeasurement(measuredAcc, measuredOmega, dt)
		cov_history.append(np.array(pim.preintMeasCov()))
		times.append(t + dt)

		if (k + 1) % int(1 / dt) == 0:
			factor = gtsam.ImuFactor(X(i), V(i), X(i + 1), V(i + 1), B(0), pim)
			graph.push_back(factor)

			# Clear preintegration
			pim.resetIntegration()

			actual_state_i = scenario.navState(t + dt)

			kf_cov = np.array(pim.preintMeasCov())
			keyframe_covs.append(kf_cov)
			keyframe_times.append(t + dt)

			rotationNoise = gtsam.Rot3.Expmap(np.random.randn(3) * init_rot_noise)
			translationNoise = gtsam.Point3(*np.random.randn(3) * init_trans_noise)
			poseNoise = gtsam.Pose3(rotationNoise, translationNoise)

			noisy_state_i = gtsam.NavState(
				actual_state_i.pose().compose(poseNoise),
				actual_state_i.velocity() + np.random.randn(3) * init_vel_noise)

			initial.insert(X(i + 1), noisy_state_i.pose())
			initial.insert(V(i + 1), noisy_state_i.velocity())
			i += 1

	if return_cov:
		return graph, initial, np.stack(cov_history), np.array(times)
	return graph, initial


def add_priors(scenario, graph, initial, actualBias):
	"""Add prior factors and initial values for pose, velocity and bias."""
	priorNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
	velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

	initial_state = scenario.navState(0)
	graph.push_back(gtsam.PriorFactorPose3(X(0), initial_state.pose(), priorNoise))
	graph.push_back(gtsam.PriorFactorVector(V(0), initial_state.velocity(), velNoise))

	initial.insert(B(0), actualBias)
	initial.insert(X(0), initial_state.pose())
	initial.insert(V(0), initial_state.velocity())

	return graph, initial


def optimize_factor_graph(graph, initial):
	"""Run batch nonlinear optimization on the factor graph."""
	params = gtsam.LevenbergMarquardtParams()
	params.setVerbosity("ERROR")
	optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
	return optimizer.optimize()


def _collect_trajectory(values):
	"""Collect pose translations from a Values object in X(i) order."""
	traj = []
	i = 0
	while values.exists(X(i)):
		traj.append(np.asarray(values.atPose3(X(i)).translation()))
		i += 1
	return np.asarray(traj) if traj else np.empty((0, 3))


def plot_trajectory(values: gtsam.Values, title: str = "Estimated Trajectory", fignum: int = 1, show: bool = False):
	"""Plot trajectory from `values` containing Pose3 entries keyed by X(i)."""
	fig = plt.figure(fignum)
	ax = fig.add_subplot(111, projection='3d')

	positions = []
	i = 0
	while values.exists(X(i)):
		pose_i = values.atPose3(X(i))
		plot_pose3(fignum, pose_i, 1)
		positions.append(np.asarray(pose_i.translation()))
		i += 1

	if positions:
		positions = np.asarray(positions)
		ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="black", linewidth=1.5, alpha=0.8)

	ax.set_title(title)

	limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
	origin = np.mean(limits, axis=1)
	radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

	ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
	ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
	ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

	if show:
		plt.show()


def plot_preintegration_data(result, show=True):
	"""Plot the pre-integrated and optimized trajectories in notebook style.

	Args:
		result: dict returned by `main(return_results=True)`.
		show: if True, call `plt.show()` at the end.

	Returns:
		A tuple of matplotlib figure objects for the initial and optimized trajectories.
	"""
	if result is None:
		raise ValueError("plot_preintegration_data expects a result dict from main(return_results=True).")

	initial = result["initial"]
	optimized = result["optimized"]

	plot_trajectory(initial, title="Initial Trajectory", fignum=1, show=False)
	plot_trajectory(optimized, title="Optimized Trajectory", fignum=2, show=False)

	fig1 = plt.figure(1)
	fig2 = plt.figure(2)

	if show:
		plt.show()
	return fig1, fig2


@dataclass
class ImuSdpEdge:
	i: int
	j: int
	R_tilde: np.ndarray
	J_bg: np.ndarray
	b_bar_g: np.ndarray
	tau_R: float
	tau_bg: float

	@property
	def R_breve(self):
		"""Shifted target rotation for linear SDP objective."""
		J_bg = np.asarray(self.J_bg).reshape(3, 3)
		b_bar = np.asarray(self.b_bar_g).reshape(3)
		return self.R_tilde @ (np.eye(3) - skew(J_bg @ b_bar))


def stamp_block(mat, row_slice, col_slice, val):
	"""Helper to stamp sub-matrices into a scipy sparse matrix."""
	r_start, r_end = row_slice
	c_start, c_end = col_slice
	val_arr = np.asarray(val)
	if val_arr.shape == ():
		val_arr = val_arr.reshape(1, 1)
	mat[r_start:r_end, c_start:c_end] = mat[r_start:r_end, c_start:c_end] + val_arr


def build_lifting_constraint(Model, name, Z_Ri_Y, Z_I3_bi, R_tilde, J_bg):
	"""Enforce the linearized skew-symmetric lift constraint for one IMU edge."""
	def skew_basis_column(vec):
		return np.array([
			[0.0, -vec[2], vec[1]],
			[vec[2], 0.0, -vec[0]],
			[-vec[1], vec[0], 0.0],
		])

	M_c = [R_tilde @ skew_basis_column(J_bg[:, k]) for k in range(3)]

	for row in range(3):
		for col in range(3):
			lhs = Z_Ri_Y.index(row, col)
			rhs = Expr.add([
				Expr.mul(M_c[0][row, col], Z_I3_bi.index(0, 0)),
				Expr.mul(M_c[1][row, col], Z_I3_bi.index(1, 0)),
				Expr.mul(M_c[2][row, col], Z_I3_bi.index(2, 0)),
			])
			Model.constraint(f"{name}_{row}_{col}", Expr.sub(lhs, rhs), Domain.equalsTo(0.0))


def build_redundant_lifting_constraints(Model, name, Z_Ri_Y, R_tilde):
	"""
	Enforce redundant skew-symmetry constraints on the lifted block:
	S = R_tilde^T * Z_{R_i, Y_{ij}}
	1. tr(S) = 0
	2. S + S^T = 0
	"""
	# S = R_tilde^T * Z_{R_i, Y_{ij}}
	S = Expr.mul(R_tilde.T, Z_Ri_Y)

	# 1. Trace constraint: tr(S) = 0
	# Fusion matrix expressions are flattened before picking diagonal entries.
	tr_S = Expr.sum(Expr.flatten(S).pick([0, 4, 8]))
	Model.constraint(f"{name}_trace", tr_S, Domain.equalsTo(0.0))

	# 2. Skew-symmetry constraint: S + S^T = 0
	S_sym = Expr.add(S, Expr.transpose(S))
	Model.constraint(f"{name}_skew", S_sym, Domain.equalsTo(0.0))


def solve_imu_sdp(num_nodes: int, edges: list, b_bar_0: np.ndarray, R_anchor: np.ndarray,
				  add_redundant: bool = True, return_extras: bool = False):
	"""Solve the block-lifting SDP relaxation for the IMU chain.

	Args:
		add_redundant: if True, include the redundant self-energy (constraint 5) and the
			redundant lifting (trace + skew) constraints. If False, only the base relaxation
			constraints are used ("no additional constraints").
		return_extras: if True, return (Z_opt, Q_dense, sdp_obj) instead of just Z_opt, where
			Q_dense is the dense objective matrix and sdp_obj = <Q, Z_opt> is the SDP optimum.
	"""
	num_edges = len(edges)

	# Matrix dimensions for the lifted SDP variable Z.
	dim_R = 3 * num_nodes
	dim_Y = 3 * num_edges
	dim_b = num_nodes
	dim_I3 = 3
	D = dim_R + dim_Y + dim_b + dim_I3

	def slice_R(i):
		return [i * 3, (i + 1) * 3]

	def slice_Y(e):
		return [dim_R + e * 3, dim_R + (e + 1) * 3]

	def slice_b(i):
		return [dim_R + dim_Y + i, dim_R + dim_Y + i + 1]

	def slice_I3():
		return [D - 3, D]

	with Model("IMU_Block_SDP") as M:
		Z = M.variable("Z", Domain.inPSDCone(D))

		Q_sparse = sp.lil_matrix((D, D), dtype=np.float64)

		for e_idx, edge in enumerate(edges):
			i, j = edge.i, edge.j
			R_breve = edge.R_breve
			stamp_block(Q_sparse, slice_Y(e_idx), slice_Y(e_idx), edge.tau_R * np.eye(3))
			stamp_block(Q_sparse, slice_R(i), slice_R(i), edge.tau_R * np.eye(3))
			stamp_block(Q_sparse, slice_R(j), slice_R(j), edge.tau_R * np.eye(3))
			stamp_block(Q_sparse, slice_R(i), slice_R(j), -edge.tau_R * R_breve)
			stamp_block(Q_sparse, slice_R(j), slice_R(i), -edge.tau_R * R_breve.T)

			# Missing Y cross-terms from the full quadratic expansion:
			# -2 tr(R_j^T Y) + 2 tr(R_breve^T R_i^T Y)
			stamp_block(Q_sparse, slice_R(j), slice_Y(e_idx), -edge.tau_R * np.eye(3))
			stamp_block(Q_sparse, slice_Y(e_idx), slice_R(j), -edge.tau_R * np.eye(3))
			stamp_block(Q_sparse, slice_R(i), slice_Y(e_idx), edge.tau_R * R_breve)
			stamp_block(Q_sparse, slice_Y(e_idx), slice_R(i), edge.tau_R * R_breve.T)

			# 1. Source-Rotation Consistency Blocks
			Z_Ri_Rj = Z.slice([slice_R(i)[0], slice_R(j)[0]], [slice_R(i)[1], slice_R(j)[1]])
			Z_Ri_Y  = Z.slice([slice_R(i)[0], slice_Y(e_idx)[0]], [slice_R(i)[1], slice_Y(e_idx)[1]])
			M.constraint(f"rlt_Ri_{e_idx}", Expr.sub(Z_Ri_Rj, Expr.add(R_breve, Z_Ri_Y)), Domain.equalsTo(0.0))

			# 2. Target-Rotation Consistency Blocks
			Z_Rj_Y  = Z.slice([slice_R(j)[0], slice_Y(e_idx)[0]], [slice_R(j)[1], slice_Y(e_idx)[1]])
			Z_Rj_Ri = Z.slice([slice_R(j)[0], slice_R(i)[0]], [slice_R(j)[1], slice_R(i)[1]])
			rhs_target = Expr.sub(np.eye(3), Expr.mul(Z_Rj_Ri, R_breve))
			M.constraint(f"rlt_Rj_{e_idx}", Expr.sub(Z_Rj_Y, rhs_target), Domain.equalsTo(0.0))

			# 3. Lifting Envelope Self-Matrix Blocks
			Z_Y_Y   = Z.slice([slice_Y(e_idx)[0], slice_Y(e_idx)[0]], [slice_Y(e_idx)[1], slice_Y(e_idx)[1]])
			Z_Y_Rj  = Z.slice([slice_Y(e_idx)[0], slice_R(j)[0]], [slice_Y(e_idx)[1], slice_R(j)[1]])
			Z_Y_Ri  = Z.slice([slice_Y(e_idx)[0], slice_R(i)[0]], [slice_Y(e_idx)[1], slice_R(i)[1]])
			rhs_lift = Expr.sub(Z_Y_Rj, Expr.mul(Z_Y_Ri, R_breve))
			M.constraint(f"rlt_Y_{e_idx}", Expr.sub(Z_Y_Y, rhs_lift), Domain.equalsTo(0.0))

			# 4. Absolute Bias Kinematic Coupling Blocks (Shape: 1 x 3)
			Z_bi_Rj = Z.slice([slice_b(i)[0], slice_R(j)[0]], [slice_b(i)[1], slice_R(j)[1]])
			Z_bi_Ri = Z.slice([slice_b(i)[0], slice_R(i)[0]], [slice_b(i)[1], slice_R(i)[1]])
			Z_bi_Y  = Z.slice([slice_b(i)[0], slice_Y(e_idx)[0]], [slice_b(i)[1], slice_Y(e_idx)[1]])
			rhs_bias = Expr.add(Expr.mul(Z_bi_Ri, R_breve), Z_bi_Y)
			M.constraint(f"rlt_bias_{e_idx}", Expr.sub(Z_bi_Rj, rhs_bias), Domain.equalsTo(0.0))

			# 5. Redundant Self-Energy Constraint on Lifting Blocks (bounds Y^T Y using rotations)
			# Z_{Y,Y} = I_3 + R_breve^T * R_breve - Z_{R_j,R_i} * R_breve - R_breve^T * Z_{R_i,R_j}
			if add_redundant:
				Z_Rj_Ri = Z.slice([slice_R(j)[0], slice_R(i)[0]], [slice_R(j)[1], slice_R(i)[1]])
				Z_Ri_Rj = Z.slice([slice_R(i)[0], slice_R(j)[0]], [slice_R(i)[1], slice_R(j)[1]])
				const_term = np.eye(3) + (R_breve.T @ R_breve)
				term2 = Expr.mul(Z_Rj_Ri, R_breve)
				term3 = Expr.mul(R_breve.T, Z_Ri_Rj)
				rhs_y_squared = Expr.sub(Expr.sub(const_term, term2), term3)
				M.constraint(f"redundant_Y_squared_{e_idx}", Expr.sub(Z_Y_Y, rhs_y_squared), Domain.equalsTo(0.0))

			# 6. Homogenized Kinematic Coupling Blocks
			Z_I3_Y  = Z.slice([slice_I3()[0], slice_Y(e_idx)[0]], [slice_I3()[1], slice_Y(e_idx)[1]])
			Z_I3_Rj = Z.slice([slice_I3()[0], slice_R(j)[0]], [slice_I3()[1], slice_R(j)[1]])
			Z_I3_Ri = Z.slice([slice_I3()[0], slice_R(i)[0]], [slice_I3()[1], slice_R(i)[1]])
			rhs_homog = Expr.sub(Z_I3_Rj, Expr.mul(Z_I3_Ri, R_breve))
			M.constraint(f"rlt_homog_{e_idx}", Expr.sub(Z_I3_Y, rhs_homog), Domain.equalsTo(0.0))

			stamp_block(Q_sparse, slice_b(i), slice_b(i), edge.tau_bg)
			stamp_block(Q_sparse, slice_b(j), slice_b(j), edge.tau_bg)
			stamp_block(Q_sparse, slice_b(i), slice_b(j), -edge.tau_bg)
			stamp_block(Q_sparse, slice_b(j), slice_b(i), -edge.tau_bg)

		Q_coo = Q_sparse.tocoo()
		Q_mosek = Matrix.sparse(D, D, Q_coo.row.astype(np.int32), Q_coo.col.astype(np.int32), Q_coo.data)
		M.objective("cost", ObjectiveSense.Minimize, Expr.dot(Q_mosek, Z))

		for i in range(num_nodes):
			Z_Ri = Z.slice([slice_R(i)[0], slice_R(i)[0]], [slice_R(i)[1], slice_R(i)[1]])
			# Enforce the lifted rotation block so that each rotation estimate satisfies R_i R_i^T = I
			M.constraint(f"ortho_R_{i}", Z_Ri, Domain.equalsTo(np.eye(3)))
			# Also enforce the transposed orthogonality constraint R_i^T R_i = I
			# M.constraint(f"ortho_RtR_{i}", Expr.transpose(Z_Ri), Domain.equalsTo(np.eye(3)))

		Z_I3_b0 = Z.slice([slice_I3()[0], slice_b(0)[0]], [slice_I3()[1], slice_b(0)[1]])
		M.constraint("bias_anchor", Z_I3_b0, Domain.equalsTo(b_bar_0.reshape(3, 1)))

		Z_I3_R0 = Z.slice([slice_I3()[0], slice_R(0)[0]], [slice_I3()[1], slice_R(0)[1]])
		M.constraint("rotation_anchor", Z_I3_R0, Domain.equalsTo(R_anchor))

		for e_idx, edge in enumerate(edges):
			i = edge.i
			Z_Ri_Y = Z.slice([slice_R(i)[0], slice_Y(e_idx)[0]], [slice_R(i)[1], slice_Y(e_idx)[1]])
			Z_I3_bi = Z.slice([slice_I3()[0], slice_b(i)[0]], [slice_I3()[1], slice_b(i)[1]])
			build_lifting_constraint(M, f"lift_{e_idx}", Z_Ri_Y, Z_I3_bi, edge.R_tilde, edge.J_bg)
			if add_redundant:
				build_redundant_lifting_constraints(M, f"lift_redundant_{e_idx}", Z_Ri_Y, edge.R_tilde)

		M.solve()
		Z_opt = np.array(Z.level()).reshape((D, D))
		if return_extras:
			Q_dense = np.asarray(Q_sparse.todense())
			sdp_obj = float(np.sum(Q_dense * Z_opt))
			return Z_opt, Q_dense, sdp_obj
		return Z_opt


def project_to_so3(M):
	"""Project an approximate 3x3 matrix onto the nearest SO(3) matrix."""
	U, _, Vt = np.linalg.svd(M)
	R = U @ Vt
	if np.linalg.det(R) < 0:
		U[:, -1] *= -1
		R = U @ Vt
	return R


def extract_sdp_solution(Z_opt, num_nodes, num_edges):
	"""Extract rotation and bias estimates from the lifted SDP matrix."""
	dim_R = 3 * num_nodes
	dim_Y = 3 * num_edges
	dim_b = num_nodes
	dim_I3 = 3
	D = dim_R + dim_Y + dim_b + dim_I3

	def slice_R(i):
		return [i * 3, (i + 1) * 3]

	def slice_b(i):
		return [dim_R + dim_Y + i, dim_R + dim_Y + i + 1]

	def slice_I3():
		return [D - 3, D]

	R_est = []
	b_est = []
	for i in range(num_nodes):
		slice_r = slice_R(i)
		R_i = Z_opt[slice_I3()[0]:slice_I3()[1], slice_r[0]:slice_r[1]]
		R_est.append(project_to_so3(R_i))
		b_i = Z_opt[slice_I3()[0]:slice_I3()[1], slice_b(i)[0]:slice_b(i)[1]].reshape(3)
		b_est.append(b_i)
	return R_est, b_est


def build_sdp_edges_from_measurements(pim_params, reference_bias, acc_measurements, gyro_measurements, dt, keyframe_interval=1.0, tau_bg=1.0):
	"""Build IMU edges from raw measurement segments for the SDP relaxation."""
	if len(acc_measurements) != len(gyro_measurements):
		raise ValueError("acc_measurements and gyro_measurements must have the same length.")

	samples_per_edge = int(round(keyframe_interval / dt))
	if samples_per_edge < 1:
		raise ValueError("keyframe_interval must be at least dt.")

	edges = []
	num_edges = len(acc_measurements) // samples_per_edge
	for e_idx in range(num_edges):
		start = e_idx * samples_per_edge
		end = start + samples_per_edge
		seg_acc = acc_measurements[start:end]
		seg_gyro = gyro_measurements[start:end]
		pim = _run_preintegration_sequence(pim_params, reference_bias, seg_acc, seg_gyro, dt)
		imu_samples = _make_imu_samples_from_measurements(seg_acc, seg_gyro, dt)
		_, dR_dbg = propagate_rotation_jacobian(imu_samples, reference_bias.gyroscope())
		R_tilde = np.asarray(pim.deltaRij().matrix())
		cov = np.asarray(pim.preintMeasCov())
		rot_var = np.mean(np.diag(cov[:3, :3])) if cov.shape[0] >= 3 else 1.0
		rot_var_floor = 1e-4
		max_tau_R = 1e8
		tau_R = min(1.0 / max(rot_var, rot_var_floor), max_tau_R)
		edges.append(ImuSdpEdge(
			i=e_idx,
			j=e_idx + 1,
			R_tilde=R_tilde,
			J_bg=dR_dbg,
			b_bar_g=np.asarray(reference_bias.gyroscope()),
			tau_R=tau_R,
			tau_bg=tau_bg,
		))
	return edges


def run_sdp_relaxation(T=10.0, dt=0.005, scenario_name="loop", keyframe_interval=1.0,
				 accel_sigma=1e-3, gyro_sigma=1e-3,
				 plot=False, return_results=False):
	"""Run the IMU preintegration scenario and solve the SDP relaxation for orientations and gyro bias."""
	scenarios = build_scenarios()
	if scenario_name not in scenarios:
		raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {sorted(scenarios.keys())}")

	angular_velocity, linear_velocity = scenarios[scenario_name]
	scenario = gtsam.ConstantTwistScenario(angular_velocity, linear_velocity)

	actual_bias, pim_params, _ = create_preintegration(
		gyro_sigma=gyro_sigma,
		accel_sigma=accel_sigma,
	)
	runner = gtsam.ScenarioRunner(scenario, pim_params, dt, actual_bias)

	acc_measurements = []
	gyro_measurements = []
	for t in np.arange(0, T, dt):
		acc_measurements.append(np.asarray(runner.measuredSpecificForce(t)))
		gyro_measurements.append(np.asarray(runner.measuredAngularVelocity(t)))

	edges = build_sdp_edges_from_measurements(
		pim_params=pim_params,
		reference_bias=actual_bias,
		acc_measurements=acc_measurements,
		gyro_measurements=gyro_measurements,
		dt=dt,
		keyframe_interval=keyframe_interval,
	)

	num_nodes = len(edges) + 1
	b_bar_0 = np.asarray(actual_bias.gyroscope())
	pose_0 = scenario.pose(0.0)
	R_anchor = np.asarray(pose_0.rotation().matrix())
	Z_opt = solve_imu_sdp(num_nodes, edges, b_bar_0, R_anchor)
	
	R_est, b_est = extract_sdp_solution(Z_opt, num_nodes, len(edges))

	# Evaluate relaxation tightness (rank) and compare per-node to ground truth
	# Rank of lifted matrix Z (low rank, e.g. 3, indicates tight relaxation)
	Z_rank = int(np.linalg.matrix_rank(Z_opt, tol=1e-8))

	# Ground-truth rotations at keyframe times
	R_true = []
	rot_errors = []
	bias_errors = []
	for k in range(num_nodes):
		time_k = k * keyframe_interval
		pose_k = scenario.pose(time_k)
		Rk = np.asarray(pose_k.rotation().matrix())
		R_true.append(Rk)
		# rotation error (angle in degrees)
		delta = R_est[k].T @ Rk
		val = (np.trace(delta) - 1.0) / 2.0
		val = np.clip(val, -1.0, 1.0)
		angle_rad = np.arccos(val)
		rot_errors.append(np.degrees(angle_rad))
		# bias error (norm)
		bias_errors.append(np.linalg.norm(b_est[k] - np.asarray(actual_bias.gyroscope())))


	result = {
		"scenario_name": scenario_name,
		"num_nodes": num_nodes,
		"keyframe_interval": keyframe_interval,
		"edges": edges,
		"Z_opt": Z_opt,
		"R_est": R_est,
		"b_est": b_est,
		"b_bar_0": b_bar_0,
		"Z_rank": Z_rank,
		"R_true": R_true,
		"rot_errors_deg": rot_errors,
		"bias_errors": bias_errors,
		"scenario": scenario,
	}

	if plot:
		plot_sdp_data(result, show=True)

	if return_results:
		return result
	return result


def plot_sdp_data(result, show=True):
	"""Plot the SDP recovered orientation trajectory and ground truth trajectory."""
	if result is None:
		raise ValueError("plot_sdp_data expects a result dict from run_sdp_relaxation(return_results=True).")

	scenario = result["scenario"]
	R_est = result["R_est"]
	num_nodes = result["num_nodes"]
	keyframe_interval = result.get("keyframe_interval", 1.0)

	fig = plt.figure(3)
	ax = fig.add_subplot(111, projection='3d')

	gt_positions = []
	head_positions = []
	for k in range(num_nodes):
		time_k = k * keyframe_interval
		pose_true = scenario.pose(time_k)
		pos_true = np.asarray(pose_true.translation())
		gt_positions.append(pos_true)

		pose_sdp = gtsam.Pose3(
			gtsam.Rot3(R_est[k]),
			gtsam.Point3(*pos_true),
		)
		head_positions.append(np.asarray(pose_sdp.translation()))

	gt_positions = np.asarray(gt_positions)
	head_positions = np.asarray(head_positions)

	ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], color='black', linewidth=2, label='Ground truth')
	ax.plot(head_positions[:, 0], head_positions[:, 1], head_positions[:, 2], color='red', linestyle='--', linewidth=2, label='SDP recovered (orientation)')

	ax.scatter(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], color='black', marker='o')
	ax.scatter(head_positions[:, 0], head_positions[:, 1], head_positions[:, 2], color='red', marker='x')

	# Plot orientation axes for ground truth and SDP estimate
	for k in range(num_nodes):
		time_k = k * keyframe_interval
		pose_true = scenario.pose(time_k)
		plot_pose3(3, pose_true, axis_length=0.25)

		pose_sdp = gtsam.Pose3(
			gtsam.Rot3(R_est[k]),
			gtsam.Point3(*gt_positions[k]),
		)
		plot_pose3(3, pose_sdp, axis_length=0.25)

	ax.set_title('SDP recovered orientations vs ground truth trajectory')
	ax.legend()

	limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
	origin = np.mean(limits, axis=1)
	radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
	ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
	ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
	ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

	if show:
		plt.show()
	return fig


def _make_imu_samples_from_measurements(acc_measurements, gyro_measurements, dt, start_timestamp_ns=0):
	"""Create lightweight IMU sample objects for analytical rotation propagation."""
	class ImuSample:
		def __init__(self, timestamp_ns, acc, gyro):
			self.timestamp_ns = int(timestamp_ns)
			self.acc_rad_s2 = np.asarray(acc)
			self.gyro_rad_s = np.asarray(gyro)

	samples = []
	timestamp_ns = int(start_timestamp_ns)
	step_ns = int(round(dt * 1e9))
	for acc, gyro in zip(acc_measurements, gyro_measurements):
		samples.append(ImuSample(timestamp_ns, acc, gyro))
		timestamp_ns += step_ns
	return samples


def main(T=10.0, dt=0.005, scenario_name="loop", return_results=False, plot=False,
	 		 accel_sigma=1e-3, gyro_sigma=1e-3,
	 		 init_rot_noise=0.1, init_trans_noise=1.0, init_vel_noise=0.1):
	"""Run the full IMU preintegration pipeline.

	This wires together:
	- scenario generation
	- preintegration setup
	- factor graph construction
	- analytical SO(3) rotation Jacobian propagation

	Args:
		T: total simulation time in seconds.
		dt: simulation timestep.
		scenario_name: key from `build_scenarios()`.
		return_results: if True, return a dict of results instead of only printing.
		plot: if True, render the summary figure after computing the result.
		accel_sigma: accelerometer noise standard deviation for measurement generation.
		gyro_sigma: gyroscope noise standard deviation for measurement generation.
		init_rot_noise: standard deviation for initial rotation guess noise.
		init_trans_noise: standard deviation for initial translation guess noise.
		init_vel_noise: standard deviation for initial velocity guess noise.

	Returns:
		Optional dict containing graph, initial, bias, preintegration data, and covariance history.
	"""
	scenarios = build_scenarios()
	if scenario_name not in scenarios:
		raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {sorted(scenarios.keys())}")

	angular_velocity, linear_velocity = scenarios[scenario_name]
	scenario = gtsam.ConstantTwistScenario(angular_velocity, linear_velocity)

	actual_bias, pim_params, pim = create_preintegration(
		gyro_sigma=gyro_sigma,
		accel_sigma=accel_sigma,
	)
	runner = gtsam.ScenarioRunner(scenario, pim_params, dt, actual_bias)
	graph = gtsam.NonlinearFactorGraph()
	initial = gtsam.Values()
	graph, initial = add_priors(scenario, graph, initial, actual_bias)

	graph, initial, cov_history, times = main_loop(
		runner=runner,
		scenario=scenario,
		pim=pim,
		graph=graph,
		initial=initial,
		T=T,
		dt=dt,
		return_cov=True,
		init_rot_noise=init_rot_noise,
		init_trans_noise=init_trans_noise,
		init_vel_noise=init_vel_noise,
	)

	# Build a measurement sequence compatible with the analytical rotation propagation.
	acc_measurements = []
	gyro_measurements = []
	for t in np.arange(0, T, dt):
		acc_measurements.append(np.asarray(runner.measuredSpecificForce(t)))
		gyro_measurements.append(np.asarray(runner.measuredAngularVelocity(t)))

	imu_samples = _make_imu_samples_from_measurements(acc_measurements, gyro_measurements, dt)
	delta_r, dR_dbg = propagate_rotation_jacobian(imu_samples, actual_bias.gyroscope())

	preint_data = extract_preintegration_data(
		pim,
		compute_bias_jacobians=True,
		pim_params=pim_params,
		bias=actual_bias,
		acc_measurements=acc_measurements,
		gyro_measurements=gyro_measurements,
		imu_samples=imu_samples,
		dt=dt,
	)
	preint_data["analytical_deltaR"] = delta_r
	preint_data["analytical_dR_dbg"] = dR_dbg

	optimized = optimize_factor_graph(graph, initial)

	# Extract per-node rotations from the optimized nonlinear solution
	R_est_main = []
	# count keyframes (poses X(i)) in the optimized values
	num_nodes_main = 0
	while optimized.exists(X(num_nodes_main)):
		num_nodes_main += 1
	for k in range(num_nodes_main):
		pose_k = optimized.atPose3(X(k))
		R_est_main.append(np.asarray(pose_k.rotation().matrix()))

	# Compare to ground truth at 1s keyframe intervals (main loop inserts factors every 1s)
	rot_errors_main = []
	for k in range(num_nodes_main):
		time_k = k * 1.0
		pose_true_k = scenario.pose(time_k)
		Rk = np.asarray(pose_true_k.rotation().matrix())
		delta = R_est_main[k].T @ Rk
		val = (np.trace(delta) - 1.0) / 2.0
		val = np.clip(val, -1.0, 1.0)
		angle_rad = np.arccos(val)
		rot_errors_main.append(np.degrees(angle_rad))

	# Bias comparison: main loop optimizes only a single bias node B(0)
	bias_error_main = None
	if optimized.exists(B(0)):
		try:
			opt_bias = optimized.atConstantBias(B(0))
			bias_error_main = np.linalg.norm(np.asarray(opt_bias.gyroscope()) - np.asarray(actual_bias.gyroscope()))
		except Exception:
			# fallback: try retrieving generic value
			try:
				opt_bias = optimized.at(B(0))
				bias_error_main = np.linalg.norm(np.asarray(opt_bias.gyroscope()) - np.asarray(actual_bias.gyroscope()))
			except Exception:
				bias_error_main = None

	result = {
		"graph": graph,
		"initial": initial,
		"optimized": optimized,
		"actual_bias": actual_bias,
		"pim_params": pim_params,
		"pim": pim,
		"preintegration": preint_data,
		"cov_history": cov_history,
		"times": times,
		"scenario_name": scenario_name,
		"scenario": scenario,
		"num_nodes_main": num_nodes_main,
		"R_est_main": R_est_main,
		"rot_errors_main_deg": rot_errors_main,
		"bias_error_main": bias_error_main,
	}

	if plot:
		plot_preintegration_data(result, show=True)

	if return_results:
		return result
	return None


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="IMU preintegration and optional SDP relaxation runner")
	parser.add_argument("--T", type=float, default=10.0, help="total simulation time (s)")
	parser.add_argument("--dt", type=float, default=0.005, help="simulation timestep (s)")
	parser.add_argument("--scenario", type=str, default="loop", help="scenario name from build_scenarios()")
	parser.add_argument("--keyframe-interval", type=float, default=1.0, help="keyframe interval for SDP edges (s)")
	parser.add_argument("--run-sdp", action="store_true", help="run the SDP relaxation after preintegration")
	parser.add_argument("--plot", action="store_true", help="show plots from the nonlinear pipeline")
	parser.add_argument("--accel-sigma", type=float, default=1e-3, help="accelerometer noise stddev for IMU measurements")
	parser.add_argument("--gyro-sigma", type=float, default=1e-3, help="gyroscope noise stddev for IMU measurements")
	parser.add_argument("--init-rot-noise", type=float, default=0.1, help="initial rotation guess noise stddev for the main nonlinear loop")
	parser.add_argument("--init-trans-noise", type=float, default=1.0, help="initial translation guess noise stddev for the main nonlinear loop")
	parser.add_argument("--init-vel-noise", type=float, default=0.1, help="initial velocity guess noise stddev for the main nonlinear loop")
	args = parser.parse_args()

	if args.run_sdp:
		print(f"Running SDP relaxation: scenario={args.scenario} T={args.T} dt={args.dt} keyframe_interval={args.keyframe_interval}")
		res = run_sdp_relaxation(
			T=args.T,
			dt=args.dt,
			scenario_name=args.scenario,
			keyframe_interval=args.keyframe_interval,
			accel_sigma=args.accel_sigma,
			gyro_sigma=args.gyro_sigma,
			plot=args.plot,
			return_results=True,
		)
		print("SDP result summary:")
		print("  num_nodes:", res["num_nodes"]) 
		print("  num_edges:", len(res["edges"]))
		print("  anchored bias b_est[0]:", res["b_est"][0])
		print("  first rotation det:", np.linalg.det(res["R_est"][0]))
		print("  Z rank:", res.get("Z_rank"))
		print("Per-node rotation error (deg) and bias error (norm):")
		for k in range(res["num_nodes"]):
			print(f"  node {k}: rot_err={res['rot_errors_deg'][k]:.4f} deg, bias_err={res['bias_errors'][k]:.6f}")
	else:
		res_main = main(
			T=args.T,
			dt=args.dt,
			scenario_name=args.scenario,
			return_results=True,
			plot=args.plot,
			accel_sigma=args.accel_sigma,
			gyro_sigma=args.gyro_sigma,
			init_rot_noise=args.init_rot_noise,
			init_trans_noise=args.init_trans_noise,
			init_vel_noise=args.init_vel_noise,
		)
		print("Nonlinear main-loop result summary:")
		print("  num_nodes:", res_main.get("num_nodes_main"))
		print("Per-node rotation error (deg):")
		for k, err in enumerate(res_main.get("rot_errors_main_deg", [])):
			print(f"  node {k}: rot_err={err:.4f} deg")
		print("Bias error at B(0):", res_main.get("bias_error_main"))