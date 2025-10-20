def get_equally_spaced_anchors_indices_recursive(initial_anchor_idx, indices, num_anchors, num_views, max_num_per_list = 7):
    max_num_per_list = max_num_per_list if num_anchors <= max_num_per_list else num_anchors
    anchor_indices = [(initial_anchor_idx + (i * round(len(indices) / num_anchors))) % len(indices) for i in range(num_anchors)]
    anchor_indices = [indices[i] for i in anchor_indices]
    
    anchor_groups = {i: [] for i in anchor_indices}
    for index_to_assign in indices:
        min_distance = float('inf')
        closest_anchor = -1
        for anchor in anchor_indices:
            direct_dist = abs(index_to_assign - anchor)
            wrap_dist = num_views - direct_dist
            distance = min(direct_dist, wrap_dist)
            if distance < min_distance:
                min_distance = distance
                closest_anchor = anchor
        anchor_groups[closest_anchor].append(index_to_assign)

    indices_list, indices_to_gen_save_flag_list = [anchor_indices], [[False] + ([True] * (len(anchor_indices)-1))]
    for anchor_idx in anchor_indices:
        indices = anchor_groups[anchor_idx]
        indices_to_gen_save = [i!=anchor_idx for i in indices]
        if any(indices_to_gen_save):
            indices_list.append(indices)
            indices_to_gen_save_flag_list.append(indices_to_gen_save)

    max_num_index_list = max([len(l) for l in indices_list])
    final_indices_list, final_indices_to_gen_save_flag_list = [], []
    if max_num_index_list > max_num_per_list:
        for indices, indices_to_gen_save_flag in zip(indices_list, indices_to_gen_save_flag_list):
            if len(indices) <= max_num_per_list:
                final_indices_list.append(indices)
                final_indices_to_gen_save_flag_list.append(indices_to_gen_save_flag)
                continue
            
            initial_anchor_idx = indices_to_gen_save_flag.index(False)
            intermediate_indices_list, intermediate_indices_to_gen_save_flag_list = get_equally_spaced_anchors_indices_recursive(
                initial_anchor_idx, indices, num_anchors, num_views, max_num_per_list=max_num_per_list)
            
            final_indices_list.extend(intermediate_indices_list)
            final_indices_to_gen_save_flag_list.extend(intermediate_indices_to_gen_save_flag_list)
        
        return final_indices_list, final_indices_to_gen_save_flag_list
    else:
        return indices_list, indices_to_gen_save_flag_list
    

def get_sweeping_anchors_indices(initial_anchor_idx, num_views):
    curr_anchors = [initial_anchor_idx, initial_anchor_idx]
    completed_indices = [initial_anchor_idx]
    indices_list, indices_to_gen_save_flag_list = [], []
    while len(completed_indices) != num_views:
        indices = list(range(curr_anchors[0]-2, curr_anchors[0]+1)) + list(range(curr_anchors[1], curr_anchors[1]+3))
        indices = list(dict.fromkeys([(i + num_views) % num_views for i in indices]))
        curr_anchors = [indices[0], indices[-1]]
        indices_to_gen_save = [i not in completed_indices for i in indices]
        completed_indices.extend([i for i in indices if i not in completed_indices])
        indices_list.append(indices)
        indices_to_gen_save_flag_list.append(indices_to_gen_save)
        
    sorted_pairs = sorted(zip(indices_list[-1], indices_to_gen_save_flag_list[-1]), key=lambda pair: pair[0], reverse=True)
    indices_list[-1], indices_to_gen_save_flag_list[-1] = zip(*sorted_pairs)
    return indices_list, indices_to_gen_save_flag_list


import graphviz
def build_and_render_tree(data, root_anchor, filename):
    dot = graphviz.Digraph(comment='Tree Structure')
    dot.attr('node', shape='circle', style='filled', fillcolor='skyblue')
    dot.attr('edge')
    dot.attr(rankdir='TB') # Arrange from Top to Bottom

    dot.node("Root", str(root_anchor), shape='ellipse')
    for view in data[0]:
        dot.node(str(view), str(view))
        dot.edge("Root", str(view))

    generated_indices = [i for i in data[0]]
    for level in data[1:]:
        parent = [i for i in level if i in generated_indices]
        assert len(parent) == 1, parent
        parent = parent[0]
        for children in level:
            if children == parent:
                continue
            dot.node(str(children), str(children))
            dot.edge(str(parent), str(children))
            generated_indices.append(children)
    dot.render(filename, format='png', view=False, cleanup=True)

import graphviz

def build_and_render_layered_graph(data, root_anchor, filename):
    dot = graphviz.Digraph(comment='Layered Graph')
    dot.attr('node', shape='circle', style='filled', fillcolor='skyblue')
    dot.attr(rankdir='LR') # Arrange from Top to Bottom

    all_nodes = set()
    for sublist in data:
        all_nodes.update(sublist)
    for node in all_nodes:
        if node == root_anchor:
            continue
        dot.node(str(node), str(node))

    dot.node("Root", str(root_anchor), shape='ellipse')
    for view in data[0]:
        if view == root_anchor:
            continue
        dot.node(str(view), str(view))
        dot.edge("Root", str(view))

    for i in range(len(data) - 1):
        current_level_nodes = set(data[i])
        next_level_nodes = set(data[i+1])

        parents = current_level_nodes.intersection(next_level_nodes)
        children = next_level_nodes.difference(current_level_nodes)
        if parents and children:
            for p in parents:
                for c in children:
                    dot.edge(str(p), str(c))

    dot.render(filename, format='png', view=False, cleanup=True)


if __name__ == "__main__":
    # Code to get the diagrams in report
    initial_anchor_idx = 0
    num_views = 20
    indices = range(num_views)
    num_anchors = 4
    max_num_per_list=4
    
    indices_list, indices_to_gen_save_flag_list = get_equally_spaced_anchors_indices_recursive(
        initial_anchor_idx, indices, num_anchors, len(indices), max_num_per_list=max_num_per_list
    )
    build_and_render_tree(indices_list, initial_anchor_idx, f'even_part_views{num_views}_anchors{num_anchors}')
    
    
    indices_list, indices_to_gen_save_flag_list = get_sweeping_anchors_indices(initial_anchor_idx, num_views)
    build_and_render_layered_graph(indices_list, initial_anchor_idx, f'sweep_part_views{num_views}_anchors{num_anchors}')