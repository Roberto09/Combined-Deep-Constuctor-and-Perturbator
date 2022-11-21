from datetime import datetime
import time
import json


def get_cur_time():
    """Returns local time as string"""
    ts = time.time()
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


class elementClass:
    def __init__(self):
        self.empty = []


def get_clean_path(arr):
    """Returns extra zeros from path.
    Dynamical model generates duplicated zeros for several graphs when obtaining partial solutions.
    """

    p1, p2 = 0, 1
    output = []

    while p2 < len(arr):

        if arr[p1] != arr[p2]:
            output.append(arr[p1])
            if p2 == len(arr) - 1:
                output.append(arr[p2])

        p1 += 1
        p2 += 1

    if output[0] != 0:
        output.insert(0, 0.0)
    if output[-1] != 0:
        output.append(0.0)

    return output


def get_journey(batch, pi, title, ind_in_batch=0):
    """Plots journey of agent

    Args:
        batch: dataset of graphs
        pi: paths of agent obtained from model
        ind_in_batch: index of graph in batch to be plotted
    """

    # Remove extra zeros
    pi_ = get_clean_path(pi[ind_in_batch].numpy())

    # Unpack variables
    depo_coord = batch[0][ind_in_batch].numpy()
    points_coords = batch[1][ind_in_batch].numpy()
    demands = batch[2][ind_in_batch].numpy()
    node_labels = [
        "(" + str(x[0]) + ", " + x[1] + ")"
        for x in enumerate(demands.round(2).astype(str))
    ]

    # Concatenate depot and points
    full_coords = np.concatenate((depo_coord.reshape(1, 2), points_coords))

    # Get list with agent loops in path
    list_of_paths = []
    cur_path = []
    for idx, node in enumerate(pi_):

        cur_path.append(node)

        if idx != 0 and node == 0:
            if cur_path[0] != 0:
                cur_path.insert(0, 0)
            list_of_paths.append(cur_path)
            cur_path = []

    list_of_path_traces = []
    for path_counter, path in enumerate(list_of_paths):
        coords = full_coords[[int(x) for x in path]]

        # Calculate length of each agent loop
        lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)

        list_of_path_traces.append(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers+lines",
                name=f"path_{path_counter}, length={total_length:.2f}",
                opacity=1.0,
            )
        )

    trace_points = go.Scatter(
        x=points_coords[:, 0],
        y=points_coords[:, 1],
        mode="markers+text",
        name="destinations",
        text=node_labels,
        textposition="top center",
        marker=dict(size=7),
        opacity=1.0,
    )

    trace_depo = go.Scatter(
        x=[depo_coord[0]],
        y=[depo_coord[1]],
        text=["1.0"],
        textposition="bottom center",
        mode="markers+text",
        marker=dict(size=15),
        name="depot",
    )

    layout = go.Layout(
        title="<b>Example: {}</b>".format(title),
        xaxis=dict(title="X coordinate"),
        yaxis=dict(title="Y coordinate"),
        showlegend=True,
        width=1000,
        height=1000,
        template="plotly_white",
    )

    data = [trace_points, trace_depo] + list_of_path_traces
    print("Current path: ", pi_)
    fig = go.Figure(data=data, layout=layout)
    fig.show()
