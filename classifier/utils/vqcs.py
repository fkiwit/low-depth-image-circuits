from abc import ABC, abstractmethod
import pennylane as qml
import jax
from jax import numpy as jnp
from jax import jit, vmap, grad
import numpy as np
import optax
import copy

jax.config.update("jax_compilation_cache_dir", "_jit_compiled")
jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_enable_x64", True)

"""
TODO:
 - Improve initialization
 - Move cost funtion to training?
 - Add target alignment?
 - Provide more circuit building blocks SU(4) etc.)
"""

# Building blocks
def cx2_ry4(params, wires):
    qml.CNOT(wires=wires)
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=wires)
    qml.RY(params[2], wires=wires[0])
    qml.RY(params[3], wires=wires[1])

def su4(params, wires):
    qml.SpecialUnitary(params, wires=wires)

class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    def setup(self):
        model = self.get_model()
        model_vmap = vmap(jit(model), in_axes=(None, 0))
        cost_fn = lambda params, state, target: optax.softmax_cross_entropy_with_integer_labels(self.temperature * model(params, state), target)
        cost_fn_vmap = vmap(jit(cost_fn), in_axes=(None, 0, 0))
        grad_fn = vmap(jit(grad(cost_fn)), in_axes=(None, 0, 0))
        return {
            "model_vmap": model_vmap,
            "params": self.params,
            "loss_fn": cost_fn_vmap,
            "grad_fn": grad_fn
        }

class NonLinearVQC(Model):
    def __init__(
              self,
              N_QUBITS,
              DEPTH,
              use_initial_state,
              building_block_tag,
              temperature,
              ):
        self.N_QUBITS = N_QUBITS
        self.use_initial_state = use_initial_state
        self.DEPTH = DEPTH
        self.temperature = temperature

        if building_block_tag == "cx2_ry4":
            self.N_PARAMS_BLOCK = 4
            self.building_block = cx2_ry4
        elif building_block_tag == "su4":
            self.N_PARAMS_BLOCK = 15
            self.building_block = su4
        else:
            raise ValueError("Building block not implemented")
        self.N_CNOT_4RY = self.N_PARAMS_BLOCK * (N_QUBITS - 1)
        self.n_params = N_QUBITS + self.N_PARAMS_BLOCK * (N_QUBITS - 1)
        self.N_PARAMS_NETWORK = self.n_params * DEPTH
        self.N_PARAMS_LINEAR = N_QUBITS * self.N_CNOT_4RY * DEPTH
        self.N_PARAMS_BIAS = self.N_CNOT_4RY * DEPTH
        self.N_LAST_LINEAR = 16 * 10 + 10

        # TODO: add more spohisticated initialization, add some hparams
        self.params = jnp.asarray(np.random.uniform(0, 1, self.N_PARAMS_NETWORK + self.N_PARAMS_LINEAR + self.N_PARAMS_BIAS + self.N_LAST_LINEAR))
        self.dev = qml.device("default.qubit", wires=self.N_QUBITS)

    def get_circuit_param_retrieval(self):
        @qml.qnode(self.dev, interface="jax")
        def circuit_param_retrieval(params, state):
            state = jax.lax.stop_gradient(state)
            qml.StatePrep(state, wires=range(self.N_QUBITS))

            counter = 0
            for i in range(self.N_QUBITS):
                qml.RY(params[i], wires=i)
                counter += 1

            for i in range(self.N_QUBITS-1):
                self.building_block(params[counter:counter+self.N_PARAMS_BLOCK], wires=[i, i+1])
                counter += self.N_PARAMS_BLOCK

            assert counter == len(params)

            return [qml.expval(qml.PauliZ(i)) for i in range(self.N_QUBITS)]

        return circuit_param_retrieval

    def get_circuit(self):
        @qml.qnode(self.dev, interface="jax")
        def circuit(params, state):
            qml.StatePrep(state, wires=range(self.N_QUBITS))

            counter = 0
            for i in range(self.N_QUBITS-1):
                self.building_block(params[counter:counter+self.N_PARAMS_BLOCK], wires=[i, i+1])
                counter += self.N_PARAMS_BLOCK

            assert counter == len(params)

            return qml.state()
        
        return circuit

    def get_get_predicition(self):

        @qml.qnode(self.dev, interface="jax")
        def get_predicition(state):
            qml.StatePrep(state, wires=range(self.N_QUBITS))
            return qml.probs(wires=range(self.N_QUBITS-4, self.N_QUBITS))

        return get_predicition

    def get_model(self):
        circuit_param_retrieval = self.get_circuit_param_retrieval()
        circuit = self.get_circuit()
        get_prediction = self.get_get_predicition()

        def model(params, state):
            if self.use_initial_state:
                initial_state = copy.deepcopy(state)
            # TODO: Add shadow models
            params_network = params[:self.N_PARAMS_NETWORK]
            params_linear = params[self.N_PARAMS_NETWORK:self.N_PARAMS_NETWORK+self.N_PARAMS_LINEAR]
            params_last_linear = params[-self.N_LAST_LINEAR:]

            for i in range(self.DEPTH):
                # Step 1: Apply non-linearity
                non_linear_params = jnp.array(circuit_param_retrieval(
                    params_network[i*self.n_params:(i+1)*self.n_params], 
                    initial_state if self.use_initial_state else state))

                # Step 2: Apply classical layer
                params_matrix = params_linear[i*(self.N_CNOT_4RY*self.N_QUBITS):(i+1)*(self.N_CNOT_4RY*self.N_QUBITS)].reshape(self.N_CNOT_4RY, self.N_QUBITS)
                non_linear_params = jnp.dot(params_matrix, non_linear_params)
                non_linear_params = jnp.tanh(non_linear_params) * jnp.pi

                # Step 3: Apply quantum circuit
                state = circuit(non_linear_params, state)

            output = get_prediction(state)
            last_linear_matrix = params_last_linear[:-10].reshape(10, 16)
            output = jnp.dot(last_linear_matrix, output)
            return output + params_last_linear[-10:]

        return model

class LinearVQC(Model):
    def __init__(
              self,
              N_QUBITS,
              DEPTH,
              building_block_tag,
              temperature,
              ):
        self.N_QUBITS = N_QUBITS
        self.DEPTH = DEPTH
        self.temperature = temperature
        self.building_block_tag = building_block_tag

        if building_block_tag == "cx2_ry4":
            self.N_PARAMS_BLOCK = 4
            self.building_block = cx2_ry4
            self.N_PARAMS_FIRST_BANK = N_QUBITS
        elif building_block_tag == "su4":
            self.N_PARAMS_BLOCK = 15
            self.building_block = su4
            self.N_PARAMS_FIRST_BANK = 0
        else:
            raise ValueError("Building block not implemented")

        self.N_PARAMS_NETWORK = self.N_PARAMS_FIRST_BANK + self.N_PARAMS_BLOCK * (N_QUBITS - 1) * DEPTH
        self.N_LAST_LINEAR = 16 * 10 + 10

        # TODO: add more spohisticated initialization, add some hparams
        self.params = jnp.asarray(np.random.normal(0, 0.01, self.N_PARAMS_NETWORK + self.N_LAST_LINEAR))
        self.dev = qml.device("default.qubit", wires=self.N_QUBITS)

    def get_circuit(self, depth):

        @qml.qnode(self.dev, interface="jax")
        def circuit(params, state):
            # TODO: Implement the right circuit
            qml.StatePrep(state, wires=range(self.N_QUBITS))

            counter = 0
            if self.building_block_tag == "cx2_ry4":
                for i in range(self.N_QUBITS):
                    qml.RY(params[i], wires=i)
                    counter += 1

            for i in range(depth):
                for j in range(self.N_QUBITS-1):
                    self.building_block(params[counter:counter+self.N_PARAMS_BLOCK], wires=[j, j+1])
                    counter += self.N_PARAMS_BLOCK

            assert counter == len(params)

            return qml.probs(wires=range(self.N_QUBITS-4, self.N_QUBITS))

        return circuit

    def get_model(self):

        circuit = self.get_circuit(self.DEPTH)
        def model(params, state):
            params_network = params[:self.N_PARAMS_NETWORK]
            params_last_linear = params[self.N_PARAMS_NETWORK:]
            output = circuit(params_network, state)
            last_linear_matrix = params_last_linear[:-10].reshape(10, 16)
            output = jnp.dot(last_linear_matrix, output)
            return output + params_last_linear[-10:]

        return model
