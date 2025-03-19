import numpy as np
import pandas as pd
import tqdm

from models import Node, Rec

checkin_file = "/home/Shared/foursquare/dataset_TSMC2014_NYC_preprocessed.csv"
df = pd.read_csv(checkin_file, encoding='ISO-8859-1')

df.columns = ["User ID", "poi", "Venue Category ID", "X", "Y", "Time"]
print("total visit :", len(df), end=' ')
df = df.drop_duplicates(subset=['poi'])
print("/ total poi :", len(df))
poi2pos = df.loc[:, ['X', 'Y', 'poi']].set_index('poi').T.to_dict('list')

id2poi = sorted(df['poi'].unique().tolist())
assert (max(id2poi) == len(id2poi) - 1) and (min(id2poi) == 0), "poi id is not continuous"

# build a tree of area
tree = Node(df['X'].min(), df['X'].max(), df['Y'].max(), df['Y'].min(), 0)
tree.build()
print("total node of tree :", Node.count)
theta = Node.theta

def main(id2poi):
    id2route = []
    id2lr = []
    id2prob = []
    max_depth = 0

    # make route/left_right_choice/probability list of each poi
    for poi in tqdm.tqdm(id2poi):
        # each poi, they have an area. p_n is each corner
        p_n = [
            (poi2pos[poi][0] - 0.5 * theta, poi2pos[poi][1] - 0.5 * theta),
            (poi2pos[poi][0] - 0.5 * theta, poi2pos[poi][1] + 0.5 * theta),
            (poi2pos[poi][0] + 0.5 * theta, poi2pos[poi][1] - 0.5 * theta),
            (poi2pos[poi][0] + 0.5 * theta, poi2pos[poi][1] + 0.5 * theta),
        ]
        
        poi_area = Rec((poi2pos[poi][1] + 0.5 * theta, poi2pos[poi][1] - 0.5 * theta,
                        poi2pos[poi][0] - 0.5 * theta, poi2pos[poi][0] + 0.5 * theta))

        route_list = []
        lr_list = []
        area_list = []
        
        for p in p_n:
            route, lr = tree.find_route(p)
            route_list.append(route)
            lr_list.append(lr)

        # remove duplicates
        route_set = list(set(tuple(r) for r in route_list))
        lr_set = list(set(tuple(l) for l in lr_list))

        # each leaf, how much they are overlapped
        for route in route_set:
            leaf_area = Rec(tree.find_idx(route[0]))
            area_list.append(leaf_area.overlap(poi_area))
        
        area_list = np.divide(area_list, sum(area_list))

        id2route.append(route_set)
        id2lr.append(lr_set)
        id2prob.append(area_list)
        max_depth = max(max_depth, max(len(x) for x in route_set))
    
    print('max_depth:', max_depth)

    max_num_routes = 4

    # Padding
    for i in range(len(id2route)):
        id2route[i] = id2route[i] + [[0]*max_depth] * (max_num_routes - len(id2route[i]))
        id2route[i] = np.array(id2route[i])

        id2lr[i] = id2lr[i] + [[0]*(max_depth-1)] * (max_num_routes - len(id2lr[i]))
        id2lr[i] = np.array(id2lr[i])

        id2prob[i] = np.append(id2prob[i], [0.0] * (max_num_routes - len(id2prob[i])))

    id2route = np.array(id2route)
    id2lr = np.array(id2lr)
    id2prob = np.array(id2prob)


    print('id2route.shape', id2route.shape)
    print('id2lr.shape', id2lr.shape)
    print('id2prob.shape', id2prob.shape)
    
    np.save(f"./npy/id2route.npy", id2route)
    np.save(f"./npy/id2lr.npy", id2lr)
    np.save(f"./npy/id2prob.npy", id2prob)
    
if __name__ == '__main__':
    main(id2poi)