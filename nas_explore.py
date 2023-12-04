import neuromorph as nm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Softmax, Embedding
from typing import Tuple, List, Callable, Union
from time import time
import os
import random

MERGE_TOKEN = "###"
MAX_LAYERS = 24
UNIT_WIDTH_STEP = 32

activation = [
    "tanh",     "sigmoid",      "relu",
    "linear",   "relu_leaky",   "relu_parametric",
    "elu",      "softmax",      "swish",        "gelu"
]

loss = [
    "mse",      "mae",              "mape",
    "huber",    "huber_modified",   "hinge"
]

convergence = [
    "multiplicative",
    "additive",
    "average"
]

weight = [
    "xavier",   "he",       "lecun",
    "uniform",  "normal"
]

bias = [
    "zero",
    "const_flat",
    "const_uneven"
]

def parametric_function(name: str, param=None) -> str:
    func = f"<{name}"
    if param:
        func += f",{param}"
    return func + ">"

def layer(name: str, size: int, activation_func=None, loss_func=None) -> str:
    node = f"({name},{size}"
    if activation_func:
        node += f",{activation_func}"
    if loss_func:
        node += f",{loss_func}"
    return node + ")\n"

def converge(name: str, convergent:str, operation: str) -> str:
    return "{" + f"{name},{convergent},{operation}" + "}\n"

def diverge(name: str, branches: List[str]) -> str:
    return f"[{name},{'|'.join([b for b in branches if b != ''])}]\n"

probabilistic = lambda x: np.random.choice(len(x), p=x)
greedy = lambda x: np.argmax(x, axis=1)[0]

@tf.function
def model_predict(m, d):
    return m(d, training=False)

def run(
    embedding_dimension: int,
    unit_width: int,
    strategy: Callable[[np.ndarray], int],
    model_shape: Tuple[int, int]
) -> Callable[[str], str]:
    tf.random.set_seed(int(time()))

    input_width, output_width = model_shape

    decision = 0
    trace = list()
    units = 0

    def token(sub: tf.keras.Model) -> int:
        decision_vector = np.array([[decision]])
        context_vector = np.array(model_predict(
            main_model,
            tf.convert_to_tensor(decision_vector, dtype=tf.float32)
        ))
        combined_input = np.concatenate([
            context_vector.flatten(),
            decision_vector.flatten()
        ]).reshape(1, 1, -1)
        prediction = model_predict(
            sub,
            tf.convert_to_tensor(combined_input, dtype=tf.float32)
        )
        probs = np.array(prediction).flatten()
        c = strategy(probs)
        trace.append(probs[c])
        sub.reset_states()
        return c

    def submodel(
        vocab_size: int,
        embedding_dim: int,
        rnn_units: int
    ) -> tf.keras.Model:
        model = Sequential()
        model.add(Dense(
            embedding_dim,
            batch_input_shape=(1, 1, embedding_dimension+1)
        ))
        model.add(LSTM(rnn_units, return_sequences=True, stateful=True))
        model.add(Dense(vocab_size, activation='softmax'))
        return model

    def mainmodel(
        embedding_dim: int,
        rnn_units: int,
        max_vocab_size: int
    ) -> tf.keras.Model:
        model = Sequential()
        model.add(Embedding(
            input_dim=max_vocab_size+1,
            output_dim=embedding_dim
        ))
        model.add(LSTM(rnn_units, return_sequences=True))
        model.add(Dense(embedding_dim, activation='relu'))
        return model

    def mergemodel(
        embedding_dim: int,
        rnn_units: int,
        max_list_length: int
    ) -> tf.keras.Model:
        # we double the vector dimension to handle maximum branch nesting
        model = Sequential()
        model.add(Embedding(
            input_dim=max_list_length*2,
            output_dim=embedding_dim
        ))
        model.add(LSTM(rnn_units, return_sequences=False))
        model.add(Dense(max_list_length*2, activation='softmax'))
        return model

    parameter_range = [i/10 for i in range(-10, 11)]
    main_model = mainmodel(embedding_dimension, unit_width, len(parameter_range))
    submodels = {
        "layer" : submodel(4,embedding_dimension,unit_width),
        "units" : submodel(len(range(0, 8)),embedding_dimension,unit_width),
        "activation" : submodel(len(activation),embedding_dimension,unit_width),
        "loss" : submodel(len(loss),embedding_dimension,unit_width),
        "parameter" : 
            submodel(len(parameter_range),embedding_dimension,unit_width),
        "branch" : submodel(2,embedding_dimension,unit_width),
        "convergence" : 
            submodel(len(convergence),embedding_dimension,unit_width)
    }
    merge_selector = mergemodel(embedding_dimension, unit_width, MAX_LAYERS)

    current_name = 0
    id = ""
    uids = list()
    dangling = list()

    def uid() -> str:
        nonlocal current_name
        nonlocal id
        id = f"{current_name:02x}"
        current_name += 1
        return id

    def determine_size() -> str:
        nonlocal decision
        decision = token(submodels["units"])
        return str(UNIT_WIDTH_STEP*(list(range(1, 9))[decision]))

    def determine_parameter() -> str:
        nonlocal decision
        decision = token(submodels["parameter"])
        return str(parameter_range[decision])

    def determine_activation() -> str:
        nonlocal decision
        decision = token(submodels["activation"])
        func = activation[decision]
        param = determine_parameter()
        return parametric_function(func, param=param)

    def determine_merger() -> str:
        nonlocal dangling
        nonlocal trace
        if len(dangling) > 0:
            end = dangling[0]
            del dangling[0]
            return end
        processed_uids = np.expand_dims(
            np.array([i for i, _ in enumerate(uids)]).reshape(-1,1),
            axis=1
        )
        predictions = np.array(model_predict(
            merge_selector,
            tf.convert_to_tensor(processed_uids, dtype=tf.float32)
        ))
        valid_length = len(uids)
        predictions[:, valid_length:] = 0 # mask probabilities
        normalized_predictions = predictions / np.sum(
            predictions,
            axis=1,
            keepdims=True
        )
        c = np.argmax(normalized_predictions, axis=1)[0]
        trace.append(normalized_predictions[0,c])
        return uids[c]

    def determine_convergence() -> str:
        nonlocal decision
        decision = token(submodels["convergence"])
        return convergence[decision]

    def determine_loss() -> str:
        nonlocal decision
        decision = token(submodels["loss"])
        func = loss[decision]
        decision = token(submodels["parameter"])
        param = determine_parameter()
        return parametric_function(func, param=param)

    def determine_branch() -> str:
        nonlocal decision
        decision = token(submodels["branch"])
        return decision

    def determine_node(branch: int) -> Tuple[str, str]:
        nonlocal decision
        nonlocal uids
        nonlocal dangling
        nonlocal id
        decision = token(submodels["layer"])
        segment = ""
        if decision == 0: # layer
            segment = layer(
                uid(),
                units,
                activation_func=determine_activation()
            )
        if decision == 1: # divergence
            branches = list()
            while determine_branch():
                branches.append(do_branch(branch+1))
            dangling.append(id)
            segment = diverge(uid(), branches)
            uids.append(id)
        if decision == 2: # convergence
            last_id = id
            segment = converge(
                uid(),
                MERGE_TOKEN,
                determine_convergence()
            )
            id = last_id
        if decision == 3: # output
            if branch:
                dangling.append(id)
                return segment, "BRANCH_END"
            return layer(
                "output",
                output_width,
                activation_func=determine_activation(),
                loss_func=determine_loss()
            ), "ACTUAL_END"
        return segment, "NEXT_NODE"

    def do_branch(branch: int) -> str:
        nonlocal dangling
        segment, maybe = determine_node(branch)
        while maybe == "NEXT_NODE":
            if len(uids) >= MAX_LAYERS-(1+branch):
                decision = 3
                if branch == 0:
                    segment += layer(
                        "output",
                        output_width,
                        activation_func=determine_activation(),
                        loss_func=determine_loss()
                    )
                break
            temp, maybe = determine_node(branch)
            segment += temp
        return segment

    def generate_mdl(header: str) -> str:
        nonlocal units
        nonlocal trace
        nonlocal uids
        nonlocal dangling
        nonlocal id
        nonlocal main_model
        nonlocal submodels
        main_model.reset_states()
        for _, sub in submodels.items():
            sub.reset_states()
        decision = 0
        units = determine_size()
        uids.clear()
        trace.clear()
        dangling.clear()
        id = ""
        mdl = f"(input, {input_width})\n"+do_branch(0)
        while mdl.find(MERGE_TOKEN) != -1:
            if len(uids) == 0:
                return generate_mdl(header)
            mdl = mdl.replace(MERGE_TOKEN, determine_merger(), 1)
        return f"{header}\n{mdl}\n"

    optimizer = tf.keras.optimizers.Adam()

    def tune_model(error: float, tape) -> None:
        nonlocal optimizer
        nonlocal main_model
        nonlocal merge_selector
        nonlocal submodels
        nonlocal trace
        reward = -error
        loss = -sum(map(tf.math.log, trace)) * reward
        trace.clear()
        def tune(sequence_model: tf.keras.Model) -> None:
            optimizer.apply_gradients(zip(
                tape.gradient(loss, sequence_model.trainable_variables),
                sequence_model.trainable_variables
            ))
        tune(main_model)
        tune(merge_selector)
        [tune(sub) for _, sub in submodels.items()]

    return generate_mdl, tune_model

def explore(input_tensor, expected_tensor, epochs):
    target_shape = (len(input_tensor[0][0]), len(expected_tensor[0][0]))
    mdl, tune = run(64, 256, probabilistic, target_shape)
    header = "/xavier,zero/"
    for i in range(100):
        with tf.GradientTape() as tape:
            description = mdl(header)
            tree = nm.compile(description, len(input_tensor[0]), 0.001)
            if tree == 0:
                tune(-100000, tape)
                continue
            candidate = nm.build(tree)
            print(f"built\n{description}\n")
            err = min([
                nm.train(candidate, input_tensor, expected_tensor, 2)
                for _ in range(epochs)
            ])
            print(f"err: {err}")
            nm.release(candidate)
            tune(err, tape)

def main():
    nm.seed(int(time()))
    generate_tensor = (lambda vector_size, batches, samples: [
            [
                [random.random() for i in range(vector_size)]
                for batch in range(batches)
            ]
            for k in range(samples)
        ]
    )
    sample_count = 1000
    batch_size = 32
    input = generate_tensor(256, batch_size, sample_count)
    expected = generate_tensor(4, batch_size, sample_count)
    explore(input, expected, 5)

if __name__=='__main__':
    main()

