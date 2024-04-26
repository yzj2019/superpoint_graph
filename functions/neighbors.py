import torch
from . import frnn



def oversample_partial_neighborhoods(neighbors, distances, k):
    """Oversample partial neighborhoods with less than k points. Missing
    neighbors are indicated by the "-1" index.

    Remarks
      - Neighbors and distances are assumed to be sorted in order of
      increasing distance
      - All neighbors are assumed to have at least one valid neighbor.
      See `search_outliers` to remove points with not enough neighbors
    """
    # Initialization
    assert neighbors.dim() == distances.dim() == 2
    device = neighbors.device

    # Get the number of found neighbors for each point. Indeed,
    # depending on the cloud properties and the chosen K and radius,
    # some points may receive `-1` neighbors
    n_found_nn = (neighbors != -1).sum(dim=1)

    # Identify points which have more than k_min and less than k
    # neighbors within R. For those, we oversample the neighbors to
    # reach k
    idx_partial = torch.where(n_found_nn < k)[0]
    neighbors_partial = neighbors[idx_partial]
    distances_partial = distances[idx_partial]

    # Since the neighbors are sorted by increasing distance, the missing
    # neighbors will always be the last ones. This helps finding their
    # number and position, for oversampling.

    # *******************************************************************
    # The above statement is actually INCORRECT because the outlier
    # removal may produce "-1" neighbors at unexpected positions. So
    # either we manage to treat this in a clean vectorized way, or we
    # fall back to the 2-searches solution...
    # Honestly, this feels like it is getting out of hand, let's keep
    # things simple, since we are not going to save so much computation
    # time with KNN wrt the partition.
    # *******************************************************************

    # For each missing neighbor, compute the size of the discrete set to
    # oversample from.
    n_valid = n_found_nn[idx_partial].repeat_interleave(
        k - n_found_nn[idx_partial])

    # Compute the oversampling row indices.
    idx_x_sampling = torch.arange(
        neighbors_partial.shape[0], device=device).repeat_interleave(
        k - n_found_nn[idx_partial])

    # Compute the oversampling column indices. The 0.9999 factor is a
    # security to handle the case where torch.rand is to close to 1.0,
    # which would yield incorrect sampling coordinates that would in
    # result in sampling '-1' indices (i.e. all we try to avoid here)
    idx_y_sampling = (n_valid * torch.rand(
        n_valid.shape[0], device=device) * 0.9999).floor().long()

    # Apply the oversampling
    idx_missing = torch.where(neighbors_partial == -1)
    neighbors_partial[idx_missing] = neighbors_partial[
        idx_x_sampling, idx_y_sampling]
    distances_partial[idx_missing] = distances_partial[
        idx_x_sampling, idx_y_sampling]

    # Restore the oversampled neighborhoods with the rest
    neighbors[idx_partial] = neighbors_partial
    distances[idx_partial] = distances_partial

    return neighbors, distances


def knn_1(
        xyz,
        k,
        r_max=1,
        batch=None,
        oversample=False,
        self_is_neighbor=False,
        verbose=False):
    """Search k-NN for a 3D point cloud xyz. This search differs
    from `knn_2` in that it operates on a single cloud input (search and
    query are the same) and it allows oversampling the neighbors when
    less than `k` neighbors are found within `r_max`. Optionally,
    passing `batch` will ensure the neighbor search does not mix up
    batch items.
    """
    assert isinstance(xyz, torch.Tensor)
    assert k >= 1
    assert xyz.dim() == 2
    assert batch is None or batch.shape[0] == xyz.shape[0]

    # To take the batch into account, we add an offset to the Z
    # coordinates. The offset is designed so that any points from two
    # batch different batch items are separated by at least `r_max + 1`
    batch_offset = 0
    if batch is not None:
        z_offset = xyz[:, 2].max() - xyz[:, 2].min() + r_max + 1
        batch_offset = torch.zeros_like(xyz)
        batch_offset[:, 2] = batch * z_offset

    # Data initialization
    device = xyz.device
    xyz_query = (xyz + batch_offset).view(1, -1, 3)
    xyz_search = (xyz + batch_offset).view(1, -1, 3)
    if not xyz.is_cuda:
        xyz_query = xyz_query.cuda()
        xyz_search = xyz_search.cuda()

    # KNN on GPU. Actual neighbor search now
    k_search = k if self_is_neighbor else k + 1
    distances, neighbors, _, _ = frnn.frnn_grid_points(
        xyz_query, xyz_search, K=k_search, r=r_max)

    # Remove each point from its own neighborhood
    neighbors = neighbors[0] if self_is_neighbor else neighbors[0][:, 1:]
    distances = distances[0] if self_is_neighbor else distances[0][:, 1:]

    # Oversample the neighborhoods where less than k points were found
    if oversample:
        neighbors, distances = oversample_partial_neighborhoods(
            neighbors, distances, k)

    # Restore the neighbors and distances to the input device
    if neighbors.device != device:
        neighbors = neighbors.to(device)
        distances = distances.to(device)

    if not verbose:
        return neighbors, distances

    # Warn the user of partial and empty neighborhoods
    num_nodes = neighbors.shape[0]
    n_missing = (neighbors < 0).sum(dim=1)
    n_partial = (n_missing > 0).sum()
    n_empty = (n_missing == k).sum()
    if n_partial == 0:
        return neighbors, distances

    print(
        f"\nWarning: {n_partial}/{num_nodes} points have partial "
        f"neighborhoods and {n_empty}/{num_nodes} have empty "
        f"neighborhoods (missing neighbors are indicated by -1 indices).")

    return neighbors, distances


def knn_2(
    x_search,
    x_query,
    k,
    r_max=1,
    batch_search=None,
    batch_query=None):
    """Search k-NN of x_query inside x_search, within radius `r_max`.
    Optionally, passing `batch_search` and `batch_query` will ensure the
    neighbor search does not mix up batch items.
    """
    assert isinstance(x_search, torch.Tensor)
    assert isinstance(x_query, torch.Tensor)
    assert k >= 1
    assert x_search.dim() == 2
    assert x_query.dim() == 2
    assert x_query.shape[1] == x_search.shape[1]
    assert bool(batch_search) == bool(batch_query)
    assert batch_search is None or batch_search.shape[0] == x_search.shape[0]
    assert batch_query is None or batch_query.shape[0] == x_query.shape[0]

    k = torch.tensor([k])
    r_max = torch.tensor([r_max])

    # To take the batch into account, we add an offset to the Z
    # coordinates. The offset is designed so that any points from two
    # batch different batch items are separated by at least `r_max + 1`
    batch_search_offset = 0
    batch_query_offset = 0
    if batch_search is not None:
        hi = max(x_search[:, 2].max(), x_query[:, 2].max())
        lo = min(x_search[:, 2].min(), x_query[:, 2].min())
        z_offset = hi - lo + r_max + 1
        batch_search_offset = torch.zeros_like(x_search)
        batch_search_offset[:, 2] = batch_search * z_offset
        batch_query_offset = torch.zeros_like(x_query)
        batch_query_offset[:, 2] = batch_query * z_offset

    # Data initialization
    device = x_search.device
    xyz_query = (x_query + batch_query_offset).view(1, -1, 3).cuda()
    xyz_search = (x_search + batch_search_offset).view(1, -1, 3).cuda()

    # KNN on GPU. Actual neighbor search now
    distances, neighbors, _, _ = frnn.frnn_grid_points(
        xyz_query, xyz_search, K=k, r=r_max)

    # Remove each point from its own neighborhood
    neighbors = neighbors[0].to(device)
    distances = distances[0].to(device)
    if k == 1:
        neighbors = neighbors[:, 0]
        distances = distances[:, 0]

    return neighbors, distances