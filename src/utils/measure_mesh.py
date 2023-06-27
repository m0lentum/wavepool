import matplotlib.collections as plt_coll
import matplotlib.patches as plt_patches
import matplotlib.pyplot as plt
import pydec


def print_measurements(cmp_complex: pydec.SimplicialComplex):
    """Measure mesh quality by examining the star_1 operator,
    print and plot results."""

    print("\nstar_1 measurements:")
    diag = cmp_complex[1].star.diagonal()
    min_val = min(diag)
    max_val = max(diag)
    print(f"value range: [{min_val}, {max_val}]")
    avg_val = sum(diag) / len(diag)
    print(f"average value: {avg_val}")
    variance = sum([(val - avg_val) ** 2 for val in diag]) / len(diag)
    print(f"variance: {variance}")
    # draw the mesh with suspicious edges highlighted
    lines = []
    colors = []
    widths = []
    for i, hodge_elem in enumerate(diag):
        edge_ends = [cmp_complex.vertices[p] for p in cmp_complex[1].simplices[i]]
        lines.append(edge_ends)
        if hodge_elem < 0.0:
            # extremely bad, likely to cause instability
            colors.append((1, 0, 0, 1))
            widths.append(4)
        elif hodge_elem < 0.2:
            # suspicious, reduces accuracy but probably does not cause instability
            colors.append((0.2, 0, 1, 1.0 - hodge_elem * 3))
            widths.append(2)
        else:
            colors.append((0, 0, 0, 0.2))
            widths.append(1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.add_collection(
        plt_coll.LineCollection(segments=lines, colors=colors, linewidths=widths)
    )
    ax.autoscale()
    red_patch = plt_patches.Patch(color=(1, 0, 0, 1), label="star_1 < 0")
    blue_patch = plt_patches.Patch(color=(0.2, 0, 1, 1), label="0 < star_1 < 0.2")
    ax.legend(handles=[red_patch, blue_patch])
    plt.show()
