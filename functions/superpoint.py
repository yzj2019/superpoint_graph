import torch
import numpy as np
from sklearn.linear_model import RANSACRegressor
import pgeof
from torch_scatter import scatter_sum, scatter_mean
from omegaconf import OmegaConf


from .utils import rgb2hsv, rgb2lab, sizes_to_pointers, to_float_rgb, \
    POINT_FEATURES, sanitize_keys, to_trimmed, isolated_nodes, available_cpu_count

from .grid_graph import edge_list_to_forward_star
from .cut_pursuit import cp_d0_dist
from .neighbors import knn_1, knn_2





class SPG:
    def __init__(self, cfg) -> None:
        self.cfg = cfg


    def Knn(self, data):
        if "batch" not in data:
            data["batch"] = None
        neighbors, distances = knn_1(
            data["pos"],
            self.cfg.Knn.k,
            r_max=self.cfg.Knn.r_max,
            batch=data["batch"]
        )
        data["neighbor_index"] = neighbors
        data["neighbor_distance"] = distances
        return data


    def GroundElevation(self, data):
        """Compute pointwise elevation by approximating the ground as a
        plane using RANSAC.

        Parameters
        ----------
        :param threshold: float
            Ground points will be searched within threshold of the lowest
            point in the cloud. Adjust this if the lowest point is below the
            ground or if you have large above-ground planar structures
        :param scale: float
            Scaling by which the computed elevation will be divided
        """
        # Recover the point positions
        pos = data["pos"].cpu().numpy()

        # To avoid capturing high above-ground flat structures, we only
        # keep points which are within `threshold` of the lowest point.
        idx_low = np.where(pos[:, 2] - pos[:, 2].min() < self.cfg.GroundElevation.threshold)[0]

        # Search the ground plane using RANSAC
        ransac = RANSACRegressor(random_state=0, residual_threshold=1e-3).fit(
            pos[idx_low, :2], pos[idx_low, 2])

        # Compute the pointwise elevation as the distance to the plane
        # and scale it
        h = pos[:, 2] - ransac.predict(pos[:, :2])
        h = h / self.cfg.GroundElevation.scale

        # elevation
        elevation = torch.from_numpy(h).to(data["pos"].device).view(-1, 1)

        data["elevation"] = elevation

        return data


    def PointFeatures(self, data):
        """Compute pointwise features based on what is already available in
        the Data object.

        All local geometric features assume the input ``Data`` has a
        ``neighbors`` attribute, holding a ``(num_nodes, k)`` tensor of
        indices. All k neighbors will be used for local geometric features
        computation, unless some are missing (indicated by -1 indices). If
        the latter, only positive indices will be used.

        The supported feature keys are the following:
          - density: local density. Assumes ``Data.neighbor_index`` and
            ``Data.neighbor_distance``
          - linearity: local linearity. Assumes ``Data.neighbor_index``
          - planarity: local planarity. Assumes ``Data.neighbor_index``
          - scattering: local scattering. Assumes ``Data.neighbor_index``
          - verticality: local verticality. Assumes ``Data.neighbor_index``
          - normal: local normal. Assumes ``Data.neighbor_index``
          - length: local length. Assumes ``Data.neighbor_index``
          - surface: local surface. Assumes ``Data.neighbor_index``
          - volume: local volume. Assumes ``Data.neighbor_index``
          - curvature: local curvature. Assumes ``Data.neighbor_index``

        :param keys: List(str)
            Features to be computed. Attributes will be saved under `<key>`
        :param k_min: int
            Minimum number of neighbors to consider for geometric features
            computation. Points with less than k_min neighbors will receive
            0-features. Assumes ``Data.neighbor_index``.
        :param k_step: int
            Step size to take when searching for the optimal neighborhood
            size following:
            http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf
            If k_step < 1, the optimal neighborhood will be computed based
            on all the neighbors available for each point.
        :param k_min_search: int
            Minimum neighborhood size used when searching the optimal
            neighborhood size. It is advised to use a value of 10 or higher.
        :param overwrite: bool
            When False, attributes of the input Data which are in `keys`
            will not be updated with the here-computed features. An
            exception to this rule is 'rgb' for which we always enforce
            [0, 1] float encoding
        """
        assert "neighbor_index" in data, \
            "Data is expected to have a 'neighbor_index' attribute"
        assert data["pos"].shape[0] < np.iinfo(np.uint32).max, \
            "Too many nodes for `uint32` indices"
        assert data["neighbor_index"].max() < np.iinfo(np.uint32).max, \
            "Too high 'neighbor_index' indices for `uint32` indices"

        # Configs
        k_min, k_step, k_min_search, overwrite = self.cfg.PointFeatures.k_min, self.cfg.PointFeatures.k_step, self.cfg.PointFeatures.k_min_search, self.cfg.PointFeatures.overwrite

        # Build the set of keys that must be computed/updated. In
        # particular, if `overwrite=False`, we do not modify
        # already-existing keys in the input Data. With the exception of
        # 'rgb', for which we always enforce [0, 1] float encoding
        keys = sanitize_keys(self.cfg.PointFeatures.keys, default=POINT_FEATURES)
        keys = set(keys) if overwrite \
            else set(keys) - set(data.keys())


        # Add local surfacic density to the features. The local density
        # is approximated as K / D² where K is the number of nearest
        # neighbors and D is the distance of the Kth neighbor. We
        # normalize by D² since points roughly lie on a 2D manifold.
        # Note that this takes into account partial neighborhoods where
        # -1 indicates absent neighbors
        if 'density' in keys:
            dmax = data["neighbor_distance"].max(dim=1).values
            k = data["neighbor_index"].ge(0).sum(dim=1)
            data["density"] = (k / dmax ** 2).view(-1, 1)

        # Add local geometric features
        needs_geof = any((
            'linearity' in keys,
            'planarity' in keys,
            'scattering' in keys,
            'verticality' in keys,
            'normal' in keys))
        if needs_geof and data["pos"] is not None:

            # Prepare data for numpy boost interface. Note: we add each
            # point to its own neighborhood before computation
            device = data["pos"].device
            xyz = data["pos"].cpu().numpy()
            nn = torch.cat(
                (torch.arange(xyz.shape[0]).view(-1, 1), data["neighbor_index"]),
                dim=1)
            k = nn.shape[1]

            # Check for missing neighbors (indicated by -1 indices)
            n_missing = (nn < 0).sum(dim=1)
            if (n_missing > 0).any():
                sizes = k - n_missing
                nn = nn[nn >= 0]
                nn_ptr = sizes_to_pointers(sizes.cpu())
            else:
                nn = nn.flatten().cpu()
                nn_ptr = torch.arange(xyz.shape[0] + 1) * k
            nn = nn.numpy().astype('uint32')
            nn_ptr = nn_ptr.numpy().astype('uint32')

            # Make sure array are contiguous before moving to C++
            xyz = np.ascontiguousarray(xyz)
            nn = np.ascontiguousarray(nn)
            nn_ptr = np.ascontiguousarray(nn_ptr)

            # C++ geometric features computation on CPU
            if k_step < 0:
                f = pgeof.compute_features(
                    xyz, 
                    nn, 
                    nn_ptr, 
                    k_min, 
                    verbose=False)
            else:
                f = pgeof.compute_features_optimal(
                    xyz,
                    nn,
                    nn_ptr,
                    k_min,
                    k_step,
                    k_min_search,
                    verbose=False)
            f = torch.from_numpy(f)

            # Keep only required features
            if 'linearity' in keys:
                data["linearity"] = f[:, 0].view(-1, 1).to(device)

            if 'planarity' in keys:
                data["planarity"] = f[:, 1].view(-1, 1).to(device)

            if 'scattering' in keys:
                data["scattering"] = f[:, 2].view(-1, 1).to(device)

            # Heuristic to increase importance of verticality in
            # partition
            if 'verticality' in keys:
                data["verticality"] = f[:, 3].view(-1, 1).to(device)
                data["verticality"] *= 2

            if 'curvature' in keys:
                data["curvature"] = f[:, 10].view(-1, 1).to(device)

            if 'length' in keys:
                data["length"] = f[:, 7].view(-1, 1).to(device)

            if 'surface' in keys:
                data["surface"] = f[:, 8].view(-1, 1).to(device)

            if 'volume' in keys:
                data["volume"] = f[:, 9].view(-1, 1).to(device)

            # As a way to "stabilize" the normals' orientation, we
            # choose to express them as oriented in the z+ half-space
            if 'normal' in keys:
                data["normal"] = f[:, 4:7].view(-1, 3).to(device)
                data["normal"][data["normal"][:, 2] < 0] *= -1

        return data
    

    def AdjacencyGraph(self, data):
        k, w = self.cfg.AdjacencyGraph.k, self.cfg.AdjacencyGraph.w
        assert "neighbor_index" in data, \
            "Data must have 'neighbor_index' attribute to allow adjacency " \
            "graph construction."
        assert "neighbor_distance" in data and data["neighbor_distance"] is not None \
               or w <= 0, \
            "Data must have 'neighbor_distance' attribute to allow adjacency " \
            "graph construction."
        assert k <= data["neighbor_index"].shape[1]

        # Compute source and target indices based on neighbors
        source = torch.arange(
            data["pos"].shape[0], device=data["pos"].device).repeat_interleave(k)
        target = data["neighbor_index"][:, :k].flatten()

        # Account for -1 neighbors and delete corresponding edges
        mask = target >= 0
        source = source[mask]
        target = target[mask]

        # Save edges and edge features in data
        data["edge_index"] = torch.stack((source, target))
        if w > 0:
            # Recover the neighbor distances and apply the masking
            distances = data["neighbor_distance"][:, :k].flatten()[mask]
            data["edge_attr"] = 1 / (w + distances / distances.mean())
        else:
            data["edge_attr"] = torch.ones_like(source, dtype=torch.float)

        return data


    def AddKeysTo(self, data, keys, to, strict=True, delete_after=True):
        """Get attributes from their keys and concatenate them to x.

        :param keys: str or list(str)
            The feature concatenated to 'to'
        :param to: str
            Destination attribute where the features in 'keys' will be
            concatenated
        :param strict: bool, optional
            Whether we want to raise an error if a key is not found
        :param delete_after: bool, optional
            Whether the Data attributes should be removed once added to 'to'
        """
        if isinstance(keys, str):
            keys = [keys]
        if keys is None or len(keys) == 0:
            return data

        for key in keys:
            # Skip if the attribute is None
            if key not in data or data[key] is None:
                if strict:
                    raise Exception(f"Data should contain the attribute '{key}'")
                else:
                    continue

            feat = data[key]
            # Remove the attribute from the Data, if required
            if delete_after:
                del data[key]

            # In case Data has no features yet
            if to not in data:
                if strict and data["pos"].shape[0] != feat.shape[0]:
                    raise Exception(f"Data should contain the attribute '{to}'")
                if feat.dim() == 1:
                    feat = feat.unsqueeze(-1)
                data[to] = feat
                continue

            x = data[to]
            # Make sure shapes match
            if x.shape[0] != feat.shape[0]:
                raise Exception(
                    f"The tensors '{to}' and '{key}' can't be concatenated, "
                    f"'{to}': {x.shape[0]}, '{key}': {feat.shape[0]}")

            # Concatenate x and feat
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            data[to] = torch.cat([x, feat], dim=-1)

        return data


    def RemoveKeys(self, data, keys, strict=False):
        """Remove attributes of a Data object based on their name.

        :param keys: str of list(str)
            List of attribute names
        :param strict: bool
            If True, will raise an exception if an attribute from key is
            not within the input Data keys
        """
        if isinstance(keys, str):
            keys = [keys]
        keys = set(keys)
        if keys is None or len(keys) == 0:
            return data
        
        for key in keys:
            if key in data:
                del data[key]
            elif strict:
                raise Exception(f"key: {key} is not within Data keys: {data.keys()}")

        return data
    

    def graph_connect_isolated(self, data, k=1):
        """Search for nodes with no edges in the graph and connect them
        to their k nearest neighbors. Update self.edge_index and
        self.edge_attr accordingly.

        Will raise an error if self has no edges or no pos.

        Returns data updated with the newly-created edges.
        """
        assert "pos" in data
        pos = data["pos"]
        device = data["pos"].device
        if "batch" not in data:
            data["batch"] = None
        batch = data["batch"]

        # Make sure there is no edge_attr if there is no edge_index
        # if not self.has_edges:
        #     self.edge_attr = None

        # self.raise_if_edge_keys()

        # Search for isolated nodes and exit if no node is isolated
        edge_index = data["edge_index"]
        is_isolated = isolated_nodes(edge_index, num_nodes=data["pos"].shape[0])
        is_out = torch.where(is_isolated)[0]
        if not is_isolated.any():
            return data

        # Search the nearest nodes for isolated nodes, among all nodes
        # NB: we remove the nodes themselves from their own neighborhood
        high = pos.max(dim=0).values
        low = pos.min(dim=0).values
        r_max = (high - low).norm()
        neighbors, distances = knn_2(
            pos,
            pos[is_out],
            k + 1,
            r_max=r_max,
            batch_search=batch,
            batch_query=batch[is_out] if batch is not None else None)
        distances = distances[:, 1:]
        neighbors = neighbors[:, 1:]

        # Add new edges between the nodes
        source = is_out.repeat_interleave(k)
        target = neighbors.flatten()
        edge_index_new = torch.vstack((source, target))
        edge_index_old = data["edge_index"]
        data["edge_index"] = torch.cat((edge_index_old, edge_index_new), dim=1)

        # Exit here if there are no edge attributes
        if "edge_attr" not in data:
            return data

        # If the edges have attributes, we also create attributes for
        # the new edges. There is no trivial way of doing so, the
        # heuristic here simply attempts to linearly regress the edge
        # weights based on the corresponding node distances.
        # First, get existing edges attributes and associated distance
        w = data["edge_attr"]
        s = edge_index_old[0]
        t = edge_index_old[1]
        d = (pos[s] - pos[t]).norm(dim=1)
        d_1 = torch.vstack((d, torch.ones_like(d))).T

        # Least square on d_1.x = w  (i.e. d.a + b = w)
        # NB: CUDA may crash trying to solve this simple system, in
        # which case we will fall back to CPU. Not ideal though
        try:
            a, b = torch.linalg.lstsq(d_1, w).solution
        except:
            print('\nWarning: torch.linalg.lstsq failed, trying again '
                    'on CPU')
            a, b = torch.linalg.lstsq(d_1.cpu(), w.cpu()).solution
            a = a.to(device)
            b = b.to(device)

        # Heuristic: linear approximation of w by d
        edge_attr_new = distances.flatten() * a + b

        # Append to existing self.edge_attr
        data["edge_attr"] = torch.cat((data["edge_attr"], edge_attr_new))

        return data


    def graph_to_trimmed(self, data, reduce='mean'):
        """Convert to 'trimmed' graph: same as coalescing with the
        additional constraint that (i, j) and (j, i) edges are duplicates.

        If edge attributes are passed, 'reduce' will indicate how to fuse
        duplicate edges' attributes.

        NB: returned edges are expressed with i<j by default.
        """
        assert "edge_index" in data and data["edge_index"].shape[1] > 0

        if "edge_attr" in data:
            edge_index, edge_attr = to_trimmed(
                data["edge_index"], edge_attr=data["edge_attr"], reduce=reduce)
        else:
            edge_index = to_trimmed(data["edge_index"])
            edge_attr = None

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr

        return data


    def CutPursuitPartition(self, data):
        """Partition Data using cut-pursuit.

        :param regularization: List(float)
        :param spatial_weight: List(float)
            Weight used to mitigate the impact of the point position in the
            partition. The larger, the less spatial coordinates matter. This
            can be loosely interpreted as the inverse of a maximum
            superpoint radius. If a list is passed, it must match the length
            of `regularization`
        :param cutoff: List(float)
            Minimum number of points in each cluster. If a list is passed,
            it must match the length of `regularization`
        :param parallel: bool
            Whether cut-pursuit should run in parallel
        :param iterations: int
            Maximum number of iterations for each partition
        :param k_adjacency: int
            When a node is isolated after a partition, we connect it to the
            nearest nodes. This rules the number of neighbors it should be
            connected to
        :param verbose: bool
        """
        # Sanity checks
        assert "edge_index" in data and data["edge_index"].shape[1] > 0, \
            "Cannot compute partition, no edges in Data"
        # assert data["pos"].shape[0] < np.iinfo(np.uint32).max, \
        #     "Too many nodes for `uint32` indices"
        # assert data["edge_index"].shape[0] < np.iinfo(np.uint32).max, \
        #     "Too many edges for `uint32` indices"

        # Initialize the hierarchical partition parameters. In particular,
        # prepare the output as list of Data objects that will be stored in
        # a NAG structure
        device = data["pos"].device
        num_threads = available_cpu_count() if self.cfg.CutPursuitPartition.parallel else 1
        data["node_size"] = torch.ones(
            data["pos"].shape[0], device=device, dtype=torch.long)  # level-0 points all have the same importance
        data_list = [data]
        # 用 OmegaConf.to_container() 将类型转换为 python 的原生 list
        regularization = OmegaConf.to_container(self.cfg.CutPursuitPartition.regularization)
        cutoff = OmegaConf.to_container(self.cfg.CutPursuitPartition.cutoff)
        spatial_weight = OmegaConf.to_container(self.cfg.CutPursuitPartition.spatial_weight)
        assert len(regularization) == len(cutoff) == len(spatial_weight)
        n_dim = data["pos"].shape[1]
        n_feat = data["x"].shape[1] if data["x"] is not None else 0

        # Iteratively run the partition on the previous partition level
        for level, (reg, cut, sw) in enumerate(zip(
                regularization, cutoff, spatial_weight)):

            if self.cfg.CutPursuitPartition.verbose:
                print(
                    f'Launching partition level={level} reg={reg}, '
                    f'cutoff={cut}')

            # Recover the Data object on which we will run the partition
            d1 = data_list[level]

            # Exit if the graph contains only one node
            if d1["pos"].shape[0] < 2:
                break

            # User warning if the number of edges exceeds uint32 limits
            # if d1["edge_index"].shape[1] > 4294967295 and self.cfg.CutPursuitPartition.verbose:
            #     print(
            #         f"WARNING: number of edges {d1["edge_index"].shape[1]} "
            #         f"exceeds the uint32 limit 4294967295. Please"
            #         f"update the cut-pursuit source code to accept a larger "
            #         f"data type for `index_t`.")

            # Convert edges to forward-star (or CSR) representation
            source_csr, target, reindex = edge_list_to_forward_star(
                d1["pos"].shape[0], d1["edge_index"].T.contiguous().cpu().numpy())
            source_csr = source_csr.astype('uint32')
            target = target.astype('uint32')
            edge_weights = d1["edge_attr"].cpu().numpy()[reindex] * reg

            # Recover attributes features from Data object
            pos_offset = d1["pos"].mean(dim=0)
            x = torch.cat((d1["pos"] - pos_offset, d1["x"]), dim=1)
            x = np.asfortranarray(x.cpu().numpy().T)
            node_size = d1["node_size"].float().cpu().numpy()
            coor_weights = np.ones(n_dim + n_feat, dtype=np.float32)
            coor_weights[:n_dim] *= sw

            # Partition computation
            super_index, x_c, cluster, edges, times = cp_d0_dist(
                n_dim + n_feat,
                x,
                source_csr,
                target,
                edge_weights=edge_weights,
                vert_weights=node_size,
                coor_weights=coor_weights,
                min_comp_weight=cut,
                cp_dif_tol=1e-2,
                cp_it_max=self.cfg.CutPursuitPartition.iterations,
                split_damp_ratio=0.7,
                verbose=self.cfg.CutPursuitPartition.verbose,
                max_num_threads=num_threads,
                balance_parallel_split=True,
                compute_Time=True,
                compute_List=True,
                compute_Graph=True)

            if self.cfg.CutPursuitPartition.verbose:
                delta_t = (times[1:] - times[:-1]).round(2)
                print(f'Level {level} iteration times: {delta_t}')
                print(f'partition {level} done')

            # Save the super_index for the i-level
            super_index = torch.from_numpy(super_index.astype('int64'))
            d1["super_index"] = super_index

            # Save cluster information in another Data object. Convert
            # cluster-to-point indices in a CSR format
            size = torch.LongTensor([c.shape[0] for c in cluster])
            # pointer = torch.cat([torch.LongTensor([0]), size.cumsum(dim=0)])
            # value = torch.cat([
            #     torch.from_numpy(x.astype('int64')) for x in cluster])
            pos = torch.from_numpy(x_c[:n_dim].T) + pos_offset.cpu()
            x = torch.from_numpy(x_c[n_dim:].T)
            s = torch.arange(edges[0].shape[0] - 1).repeat_interleave(
                torch.from_numpy((edges[0][1:] - edges[0][:-1]).astype("int64")))
            t = torch.from_numpy(edges[1].astype("int64"))
            edge_index = torch.vstack((s, t))
            edge_attr = torch.from_numpy(edges[2] / reg)
            node_size = torch.from_numpy(node_size)
            node_size_new = scatter_sum(
                node_size.cuda(), super_index.cuda(), dim=0).cpu().long()
            d2 = dict(
                pos=pos, x=x, edge_index=edge_index, edge_attr=edge_attr,
                # sub=Cluster(pointer, value), 
                node_size=node_size_new)

            # Merge the lower level's instance annotations, if any
            # if d1.obj is not None and isinstance(d1.obj, InstanceData):
            #     d2.obj = d1.obj.merge(d1.super_index)

            # Trim the graph
            d2 = self.graph_to_trimmed(d2)

            # If some nodes are isolated in the graph, connect them to
            # their nearest neighbors, so their absence of connectivity
            # does not "pollute" higher levels of partition
            if d2["pos"].shape[0] > 1:
                d2 = self.graph_connect_isolated(d2, k=self.cfg.CutPursuitPartition.k_adjacency)

            # Aggregate some point attributes into the clusters. This
            # is not performed dynamically since not all attributes can
            # be aggregated (e.g. 'neighbor_index', 'neighbor_distance',
            # 'edge_index', 'edge_attr'...)
            if 'y' in d1:
                assert d1["y"].dim() == 2, \
                    "Expected Data.y to hold `(num_nodes, num_classes)` " \
                    "histograms, not single labels"
                d2.y = scatter_sum(
                    d1["y"].cuda(), d1["super_index"].cuda(), dim=0).cpu()
                torch.cuda.empty_cache()

            # if 'semantic_pred' in d1.keys:
            #     assert d1.semantic_pred.dim() == 2, \
            #         "Expected Data.semantic_pred to hold `(num_nodes, num_classes)` " \
            #         "histograms, not single labels"
            #     d2.semantic_pred = scatter_sum(
            #         d1.semantic_pred.cuda(), d1.super_index.cuda(), dim=0).cpu()
            #     torch.cuda.empty_cache()

            # TODO: aggregate other attributes ?

            # TODO: if scatter operations are bottleneck, use scatter_csr

            # Add the l+1-level Data object to data_list and update the
            # l-level after super_index has been changed
            data_list[level] = d1
            data_list.append(d2)

            if self.cfg.CutPursuitPartition.verbose:
                print('\n' + '-' * 64 + '\n')

        return data_list


    def segment_points(self, coords, colors):
        '''
        points or voxels as xyz rgb
        '''
        data = dict(pos=coords, rgb=colors)
        data = self.Knn(data)
        data = self.GroundElevation(data)
        data = self.PointFeatures(data)
        data = self.AdjacencyGraph(data)
        data = self.graph_connect_isolated(data)
        data = self.AddKeysTo(data, self.cfg.partition_hf, "x")
        # Trim the graph
        # TODO: calling this on the level-0 adjacency graph is a bit sluggish
        #  but still saves partition time overall. May be worth finding a
        #  quick way of removing self loops and redundant edges...
        data = self.graph_to_trimmed(data)

        data_list = self.CutPursuitPartition(data)
        return data_list
        # data = self.RemoveKeys(data, "x", strict=True)